"""Local human-annotation UI. Only explicit form saves write labels; no AI or API calls."""
import argparse
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

from phone_assistant.common.image_mime import image_mime
from phone_assistant.recommendation.recsys_data import ROOT, read_jsonl

SCHEMAS = {
    "retrieval": ('artifacts/evaluation/handset_human_review_v1/retrieval_labels.csv', ("manual_relevance", "reviewer", "notes"), "manual_relevance"),
    "answers": ('artifacts/evaluation/handset_human_review_v1/answer_labels.csv', ("manual_faithfulness", "manual_relevance", "manual_citation_correctness", "reviewer", "notes"), "manual_faithfulness"),
    "ram": ('artifacts/evaluation/manual_handset_suspicious_ram.csv', ("manual_correct_ram_gb", "reviewer", "notes"), "manual_correct_ram_gb"),
    "catalog": ('artifacts/evaluation/manual_handset_catalog_labels.csv', ("human_product_type", "metadata_issue", "reviewer", "notes"), "human_product_type"),
}


class Conflict(ValueError):
    pass


class ReviewStore:
    def __init__(self, root=ROOT):
        self.root, self.lock = Path(root), threading.RLock()

    def read(self, kind):
        if not isinstance(kind, str) or kind not in SCHEMAS:
            raise ValueError("Unknown review kind")
        path = self.root / SCHEMAS[kind][0]
        content = path.read_bytes()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        rows = list(reader)
        if not set(SCHEMAS[kind][1]) <= set(reader.fieldnames or []):
            raise ValueError("Review sheet schema mismatch")
        return path, reader.fieldnames, rows, hashlib.sha256(content).hexdigest(), content

    def task(self, kind, index):
        with self.lock:
            _, _, rows, version, _ = self.read(kind)
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError("Review index out of bounds")
            key = SCHEMAS[kind][2]
            values = [row.get(key, "").strip() for row in rows]
            if kind == "answers":
                values = []
                for row in rows:
                    grades = [row.get(k, "").strip() for k in SCHEMAS[kind][1] if k.startswith("manual_")]
                    values.append("complete" if all(v in ("0", "1", "2") for v in grades) else
                        "uncertain" if "uncertain" in grades else "partial" if any(grades) else "")
            return {"kind": kind, "index": index, "row": rows[index], "editable": SCHEMAS[kind][1], "version": version,
                "progress": {"total": len(rows), "decided": sum(v not in ("", "uncertain", "partial") for v in values),
                    "uncertain": values.count("uncertain"), "blank": values.count(""), "partial": values.count("partial")}}

    def save(self, kind, index, changes, version):
        with self.lock:
            path, fields, rows, current, content = self.read(kind)
            if version != current:
                raise Conflict("File changed since loading. Refresh before saving; other edits were retained.")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rows):
                raise ValueError("Review index out of bounds")
            if not isinstance(changes, dict) or not set(changes) <= set(SCHEMAS[kind][1]):
                raise ValueError("Only manual label/reviewer/notes fields can be edited")
            if any(not isinstance(v, str) or len(v) > 4000 for v in changes.values()):
                raise ValueError("Invalid review value")
            updated = {**rows[index], **{k: v.strip() for k, v in changes.items()}}
            graded = [updated.get(k, "") for k in SCHEMAS[kind][1] if k.startswith("manual_") and k != "manual_correct_ram_gb"]
            if any(v not in ("", "uncertain", "0", "1", "2") for v in graded):
                raise ValueError("Grades must be 0/1/2, uncertain or blank")
            if kind == "answers" and not updated.get("raw_answer", "").strip() and any(v not in ("", "uncertain") for v in graded):
                raise ValueError("Missing/failed generation cannot receive a quality grade")
            label = updated.get(SCHEMAS[kind][2], "")
            if kind == "catalog" and label not in ("", "handset", "accessory", "other", "uncertain"):
                raise ValueError("Invalid product-type label")
            if kind == "ram" and label:
                try:
                    value = float(label)
                except ValueError as exc:
                    raise ValueError("RAM correction must be positive GB") from exc
                if not math.isfinite(value) or value <= 0 or not updated.get("notes"):
                    raise ValueError("RAM correction needs a positive value and source/evidence in notes")
            if (label not in ("", "uncertain") or any(v not in ("", "uncertain") for v in graded)) and not updated.get("reviewer"):
                raise ValueError("A decided label needs a reviewer name/code")
            backup = self.root / 'artifacts/evaluation/review_backups' / (kind + ".initial.csv")
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                backup.write_bytes(content)
            old = rows[index]
            rows[index] = updated
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
            encoded = buffer.getvalue().encode("utf-8-sig")
            temporary = path.with_suffix(path.suffix + ".review.tmp")
            temporary.write_bytes(encoded)
            # Catch external spreadsheet edits which happened during validation/preparation.
            if hashlib.sha256(path.read_bytes()).hexdigest() != current:
                temporary.unlink()
                raise Conflict("External edits detected; refresh before saving")
            temporary.replace(path)
            journal = {"kind": kind, "index": index, "source_version": current, "new_version": hashlib.sha256(encoded).hexdigest(),
                "identity": {key: updated[key] for key in ("query_id", "item_id") if key in updated},
                "before": {k: old.get(k) for k in SCHEMAS[kind][1]}, "after": {k: updated.get(k) for k in SCHEMAS[kind][1]}}
            with (backup.parent / "changes.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(journal, ensure_ascii=False) + "\n")
            return self.task(kind, index)


HTML = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Đánh giá dữ liệu điện thoại</title>
<style>body{font:16px Arial;max-width:1050px;margin:24px auto;padding:0 16px;line-height:1.5}button,select,input,textarea{font:inherit;padding:8px;margin:5px}textarea{width:95%;min-height:72px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:14px}img{max-width:220px;max-height:220px;float:right}label{display:block}#message{color:#9b270c}</style>
<h1>Đánh giá dữ liệu điện thoại</h1><p>Không dùng API. Nhãn chỉ được lưu khi bạn bấm Lưu. Ô trống/uncertain là chưa có kết luận; không tự sửa thông số theo phỏng đoán.</p>
<select id="kind"><option value="retrieval">Độ liên quan sản phẩm</option><option value="answers">Câu trả lời đã sinh</option><option value="ram">RAM cần xác minh</option><option value="catalog">Loại sản phẩm</option></select>
<button id="prev">Trước</button><button id="next">Tiếp</button><button id="skip">Đến dòng chưa chấm</button><span id="progress"></span>
<p>Relevance: 2=đáp ứng, 1=một phần, 0=không đáp ứng. Answer: 0–2 cho từng tiêu chí. RAM: nhập GB khi có nguồn xác minh, ghi nguồn trong notes. Reviewer có thể dùng mã R1.</p>
<div id="message"></div><img id="photo" hidden><pre id="row"></pre><form id="form"></form><button id="save">Lưu nhãn</button>
<script>
let current=null,index=0;const kind=document.getElementById('kind'),form=document.getElementById('form'),message=document.getElementById('message');form.onsubmit=e=>e.preventDefault();document.getElementById('photo').onerror=e=>e.target.hidden=true;
async function load(){message.textContent='';const r=await fetch('/api/task?kind='+kind.value+'&index='+index),v=await r.json();if(!r.ok){message.textContent=v.error;return}current=v;
document.getElementById('progress').textContent=(index+1)+'/'+v.progress.total+' · Đã kết luận '+v.progress.decided+' · Trống '+v.progress.blank+' · Dở dang '+v.progress.partial+' · Chưa rõ '+v.progress.uncertain;
const row={...v.row};v.editable.forEach(k=>delete row[k]);document.getElementById('row').textContent=JSON.stringify(row,null,2);form.replaceChildren();
for(const key of v.editable){const label=document.createElement('label');label.textContent=key;const input=document.createElement(key==='notes'?'textarea':'input');input.name=key;input.value=v.row[key]||'';label.appendChild(input);form.appendChild(label)}
const photo=document.getElementById('photo');const id=v.row.item_id;photo.hidden=!id;if(id)photo.src='/image?item_id='+encodeURIComponent(id);}
document.getElementById('prev').onclick=()=>{if(index>0){index--;load()}};document.getElementById('next').onclick=()=>{if(current&&index+1<current.progress.total){index++;load()}};kind.onchange=()=>{index=0;load()};
document.getElementById('skip').onclick=async()=>{if(!current)return;for(let i=1;i<=current.progress.total;i++){let j=(index+i)%current.progress.total;const r=await fetch('/api/task?kind='+kind.value+'&index='+j);const v=await r.json();const key=({retrieval:'manual_relevance',answers:'manual_faithfulness',ram:'manual_correct_ram_gb',catalog:'human_product_type'})[kind.value];if(kind.value==='answers'?v.editable.filter(k=>k.startsWith('manual_')).some(k=>!v.row[k]||v.row[k]==='uncertain'):!v.row[key]){index=j;load();return}}message.textContent='Không còn dòng trống.'};
document.getElementById('save').onclick=async()=>{const changes=Object.fromEntries(new FormData(form));const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:kind.value,index,changes,version:current.version})});const v=await r.json();if(!r.ok){message.textContent=v.error;return}current=v;await load();message.textContent='Đã lưu nhãn của bạn.'};load();
</script></html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    store = None
    documents = {}

    def send_json(self, value, status=200):
        content = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            content = HTML.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(content)
        elif parsed.path == "/api/task":
            query = parse_qs(parsed.query)
            try:
                self.send_json(self.store.task(query.get("kind", [""])[0], int(query.get("index", ["0"])[0])))
            except (ValueError, OSError) as exc:
                self.send_json({"error": str(exc)}, 400)
        elif parsed.path == "/image":
            item = parse_qs(parsed.query).get("item_id", [""])[0]
            paths = self.documents.get(item, {}).get("product_image_paths", [])
            if not paths or not Path(paths[0]).is_file():
                self.send_error(404); return
            path = Path(paths[0]); data = path.read_bytes()
            self.send_response(200); self.send_header("Content-Type", image_mime(path)); self.end_headers(); self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            self.send_error(404); return
        host, port = self.server.server_address
        if self.headers.get("Origin") not in (None, f"http://{host}:{port}", f"http://localhost:{port}"):
            self.send_json({"error": "Different origin"}, 403); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise ValueError("Invalid request size")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("Expected an object")
            self.send_json(self.store.save(value.get("kind"), value.get("index"), value.get("changes"), value.get("version")))
        except Conflict as exc:
            self.send_json({"error": str(exc)}, 409)
        except (ValueError, OSError) as exc:
            self.send_json({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    ReviewHandler.store = ReviewStore()
    ReviewHandler.documents = {d["doc_id"]: d for d in read_jsonl(ROOT / 'artifacts/cache/handset_runtime_v1/documents.jsonl')}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ReviewHandler)
    print(f"Human review: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
