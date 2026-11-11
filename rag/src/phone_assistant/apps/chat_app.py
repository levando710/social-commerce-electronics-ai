"""Small local chat UI for the RAG demo."""
from phone_assistant.common.project_layout import MODULE_ROOT
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie, CookieError
from phone_assistant.common.project_layout import asset_path
import argparse
import json
from phone_assistant.common.image_mime import image_mime
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from phone_assistant.retrieval.serving.retrieval_service import METHODS, retrieve
from phone_assistant.conversation.conversation_state import SessionStore
from phone_assistant.conversation.catalog_answer import answer_policy


ROOT = MODULE_ROOT
HTML = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Phone Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef2f7;
      --panel: #ffffff;
      --line: #d8dee9;
      --text: #172033;
      --muted: #667085;
      --blue: #0866ff;
      --blue-soft: #e8f1ff;
      --green: #128c7e;
      --shadow: 0 18px 45px rgba(15, 23, 42, .14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: grid;
      place-items: center;
      padding: 24px;
    }
    .phone {
      width: min(980px, 100%);
      height: min(760px, calc(100vh - 48px));
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
    }
    header {
      height: 64px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 18px;
      background: #fff;
    }
    .avatar {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      background: var(--blue);
      display: grid;
      place-items: center;
      color: white;
      font-weight: 700;
    }
    h1 {
      font-size: 16px;
      margin: 0;
      line-height: 1.25;
    }
    .status {
      font-size: 12px;
      color: var(--muted);
      margin-top: 2px;
    }
    main {
      overflow: auto;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      background: linear-gradient(#f8fafc, #eef2f7);
    }
    .row {
      display: flex;
      width: 100%;
    }
    .row.user { justify-content: flex-end; }
    .bubble {
      max-width: min(760px, 86%);
      border-radius: 8px;
      padding: 11px 13px;
      line-height: 1.45;
      font-size: 14px;
      white-space: pre-wrap;
    }
    .assistant .bubble {
      background: #fff;
      border: 1px solid var(--line);
    }
    .user .bubble {
      background: var(--blue);
      color: #fff;
    }
    .cards {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 12px;
    }
    .card img {
      width: 92px;
      height: 92px;
      border-radius: 6px;
      object-fit: contain;
      border: 1px solid #edf0f5;
      background: #fff;
    }
    .placeholder {
      width: 92px;
      height: 92px;
      border-radius: 6px;
      background: #eef2f7;
      border: 1px solid #edf0f5;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .title {
      font-weight: 700;
      margin-bottom: 5px;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 7px;
    }
    .evidence {
      color: #344054;
      font-size: 13px;
    }
    .answer {
      border-left: 3px solid var(--blue);
      padding-left: 10px;
      margin-top: 8px;
      color: #172033;
    }
    .llm-note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    form {
      border-top: 1px solid var(--line);
      background: #fff;
      padding: 12px;
      display: flex;
      gap: 10px;
    }
    input {
      flex: 1;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 12px 16px;
      font-size: 14px;
      outline: none;
      background: #f8fafc;
    }
    input:focus {
      border-color: var(--blue);
      background: #fff;
    }
    button {
      border: 0;
      background: var(--blue);
      color: #fff;
      min-width: 46px;
      border-radius: 999px;
      font-size: 18px;
      cursor: pointer;
    }
    button:disabled {
      opacity: .55;
      cursor: wait;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .chip {
      border: 1px solid var(--line);
      background: var(--blue-soft);
      color: #1849a9;
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 12px;
      cursor: pointer;
    }
    @media (max-width: 720px) {
      body { padding: 0; }
      .phone { height: 100vh; border-radius: 0; border: 0; }
      .card { grid-template-columns: 72px 1fr; }
      .card img, .placeholder { width: 72px; height: 72px; }
      .bubble { max-width: 94%; }
    }
  </style>
</head>
<body data-history="0">
  <section class="phone">
    <header>
      <div class="avatar">AI</div>
      <div>
        <h1>Demo Multimodal RAG điện thoại</h1>
        <div class="status">Dữ liệu và ảnh sản phẩm · __API_MODE__</div>
      </div>
      <div style="margin-left:auto;display:flex;gap:8px;align-items:center">
        <input id="profile" type="number" min="0" placeholder="Hồ sơ thử nghiệm" style="width:150px;padding:8px;border-radius:6px;font-size:12px" title="Để trống để dùng gợi ý phổ biến" />
        <button id="new-chat" type="button" style="font-size:12px;padding:8px;border-radius:6px">Hội thoại mới</button>
        <button id="clear-profile" type="button" hidden style="font-size:12px;padding:8px;border-radius:6px">Xóa sở thích</button>
      </div>
    </header>
    <main id="chat">
      <div class="row assistant">
        <div class="bubble">
          Nhập câu hỏi về điện thoại. Ví dụ:
          <div class="chips">
            <span class="chip">iphone unlocked renewed good camera</span>
            <span class="chip">samsung galaxy unlocked good battery</span>
            <span class="chip">cheap nokia basic phone</span>
          </div>
        </div>
      </div>
    </main>
    <form id="form">
      <input id="query" autocomplete="off" placeholder="Nhập câu hỏi..." />
      <button id="send" title="Gửi">➤</button>
    </form>
  </section>
  <script>
    const chat = document.querySelector("#chat");
    const form = document.querySelector("#form");
    const query = document.querySelector("#query");
    const send = document.querySelector("#send");
    const profile = document.querySelector("#profile");
    const historyEnabled = document.body.dataset.history === "1";
    profile.hidden = historyEnabled;
    document.querySelector("#clear-profile").hidden = !historyEnabled;
    document.querySelector("#clear-profile").addEventListener("click", async () => {
      if (send.disabled) return;
      if (!confirm("Xóa lịch sử tìm kiếm và các lựa chọn đã ghi trên trình duyệt này?")) return;
      try {
        const response = await fetch("/api/profile/reset", {method:"POST", headers:{"Content-Type":"application/json"}, body:"{}"});
        if (!response.ok) throw new Error("Không xóa được lịch sử");
        sessionId = null;
        chat.replaceChildren();
        addBubble("assistant", "Đã xóa lịch sử và sở thích.");
      } catch (error) { addBubble("assistant", error.message); }
    });
    let sessionId = null;
    document.querySelector("#new-chat").addEventListener("click", () => {
      if (send.disabled) return;
      sessionId = null;
      chat.replaceChildren();
      profile.disabled = false;
      addBubble("assistant", "Đã bắt đầu hội thoại mới.");
    });

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c]));
    }

    function addBubble(role, content) {
      const row = document.createElement("div");
      row.className = `row ${role}`;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      if (typeof content === "string") bubble.textContent = content;
      else bubble.appendChild(content);
      row.appendChild(bubble);
      chat.appendChild(row);
      chat.scrollTop = chat.scrollHeight;
      return row;
    }

    function resultView(data) {
      const wrap = document.createElement("div");
      const intro = document.createElement("div");
      intro.textContent = data.results.length
        ? `Mình tìm thấy ${data.results.length} kết quả liên quan nhất:`
        : "Chưa tìm thấy kết quả phù hợp.";
      wrap.appendChild(intro);
      if (data.answer) {
        if (data.answer_mode && data.answer_mode.startsWith("catalog_template")) {
          const label = document.createElement("div");
          label.className = "llm-note";
          label.textContent = "Trả lời từ dữ liệu sản phẩm, không dùng AI sinh văn bản.";
          wrap.appendChild(label);
        }
        const answer = document.createElement("div");
        answer.className = "answer";
        answer.textContent = data.answer;
        wrap.appendChild(answer);
      } else if (data.llm_error) {
        const note = document.createElement("div");
        note.className = "llm-note";
        note.textContent = "Gemini chưa trả lời: " + data.llm_error;
        wrap.appendChild(note);
      }
      if (data.query_plan && !historyEnabled) {
        const plan = document.createElement("div");
        plan.className = "llm-note";
        const tags = (data.query_plan.tags || []).join(", ");
        plan.textContent = `Query understanding (${data.query_plan.source}): ${data.query_plan.query_rewrite || data.query} ${tags ? " | tags: " + tags : ""}`;
        wrap.appendChild(plan);
      }
      const cards = document.createElement("div");
      cards.className = "cards";
      for (const item of data.results) {
        const card = document.createElement("div");
        card.className = "card";
        const image = item.product_images?.[0]
          ? `<img src="/image?path=${encodeURIComponent(item.product_images[0])}" alt="">`
          : `<div class="placeholder">No image</div>`;
        const scoreLabel = historyEnabled ? "" : `Điểm ${item.score} · `;
        const specNotice = (item.catalog_facts?.fact_warnings || []).includes("ram_capacity_requires_review")
          ? '<div class="llm-note">Thông số RAM trong nguồn dữ liệu cần kiểm tra lại; chưa được xác minh.</div>' : '';
        card.innerHTML = `
          ${image}
          <div>
            <div class="title">${escapeHtml(item.title)}</div>
            <div class="meta">${escapeHtml(scoreLabel)}Rating ${escapeHtml(item.average_rating)} · ${escapeHtml(item.review_count)} review · ${escapeHtml(item.brand)}</div>
            <div class="evidence">${escapeHtml(item.evidence)}</div>
            ${specNotice}
          </div>`;
        if (historyEnabled) {
          const reviews = document.createElement("details");
          const heading = document.createElement("summary");
          heading.textContent = "Ý kiến người dùng công khai";
          const body = document.createElement("div");
          body.className = "evidence";
          reviews.append(heading, body);
          let loaded = false;
          reviews.addEventListener("toggle", async () => {
            if (!reviews.open || loaded) return;
            body.textContent = "Đang tải...";
            try {
              const response = await fetch('/api/reviews?item_id=' + encodeURIComponent(item.doc_id));
              const result = await response.json();
              if (!response.ok) throw new Error(result.error);
              body.textContent = result.notice + '\n' + (result.reviews.length
                ? result.reviews.map(r => '[' + r.doc_id + '] ' + r.title + '\n' + r.text).join('\n\n')
                : 'Chưa có review phù hợp trong nguồn train.');
              body.style.whiteSpace = "pre-wrap";
              loaded = true;
            } catch (error) { body.textContent = error.message; }
          });
          card.lastElementChild.appendChild(reviews);
        }
        if (historyEnabled) {
          const actions = document.createElement("div");
          actions.style.cssText = "display:flex;gap:8px;margin-top:8px;flex-wrap:wrap";
          const selected = data.profile_actions?.[item.doc_id] || {};
          for (const [kind, label, activeLabel, undo] of [
            ["view", "Đánh dấu đã xem", "Đã xem", null],
            ["like", "Thích", "Bỏ thích", "unlike"],
            ["save", "Lưu", "Bỏ lưu", "unsave"]]) {
            const button = document.createElement("button");
            button.type = "button";
            button.style.cssText = "font-size:12px;padding:6px 10px;border-radius:6px";
            let active = Boolean(selected[kind]);
            button.textContent = active ? activeLabel : label;
            button.disabled = active && !undo;
            button.addEventListener("click", async () => {
              button.disabled = true;
              try {
                const response = await fetch("/api/events", {method:"POST", headers:{"Content-Type":"application/json"},
                  body:JSON.stringify({kind:active && undo ? undo : kind, item_id:item.doc_id, event_id:crypto.randomUUID()})});
                const result = await response.json();
                if (!response.ok) throw new Error(result.error || "Không lưu được lựa chọn");
                active = Boolean(result.profile.active_actions?.[item.doc_id]?.[kind]);
                button.textContent = active ? activeLabel : label;
              } catch (error) { addBubble("assistant", error.message); }
              finally { button.disabled = active && !undo; }
            });
            actions.appendChild(button);
          }
          card.lastElementChild.appendChild(actions);
        }
        cards.appendChild(card);
      }
      wrap.appendChild(cards);
      if (data.recommendations?.length) {
        const feed = document.createElement("div");
        feed.className = "answer";
        const title = document.createElement("strong");
        title.textContent = historyEnabled ? "Gợi ý cho bạn" : "Gợi ý theo nhu cầu hiện tại";
        feed.appendChild(title);
        for (const item of data.recommendations) {
          const line = document.createElement("div");
          line.textContent = item.title;
          feed.appendChild(line);
        }
        wrap.appendChild(feed);
      }
      return wrap;
    }

    async function ask(text) {
      addBubble("user", text);
      query.value = "";
      send.disabled = true;
      const loading = addBubble("assistant", "Đang truy xuất dữ liệu và gọi Gemini...");
      try {
        const response = await fetch("/api/query", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query: text, top_k: 5, use_llm: true, use_query_llm: false,
            session_id: sessionId, user_index: profile.value === "" ? null : Number(profile.value)})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Truy xuất thất bại");
        sessionId = data.session_id || null;
        profile.disabled = Boolean(sessionId);
        loading.remove();
        addBubble("assistant", resultView(data));
      } catch (error) {
        loading.remove();
        addBubble("assistant", "Có lỗi khi truy vấn: " + error.message);
      } finally {
        send.disabled = false;
        query.focus();
      }
    }

    form.addEventListener("submit", event => {
      event.preventDefault();
      const text = query.value.trim();
      if (text) ask(text);
    });
    document.querySelectorAll(".chip").forEach(chip => {
      chip.addEventListener("click", () => ask(chip.textContent.trim()));
    });
    query.focus();
  </script>
</body>
</html>
"""


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def best_snippet(text, query_terms, width=420):
    lower = text.lower()
    positions = [lower.find(term) for term in query_terms if lower.find(term) >= 0]
    start = max(min(positions) - 120, 0) if positions else 0
    snippet = text[start:start + width].replace("\n", " ")
    return snippet[:width].strip()


class ChatServer(BaseHTTPRequestHandler):
    documents = []
    by_id = {}
    index = {}
    dense = None
    method = "legacy"
    recommender = None
    co_guiding = None
    coupling_mode = "coupled"
    sessions = SessionStore()
    interaction_store = None
    local_preferences = None
    llm_allowed = False
    catalog_answers = True
    review_evidence = None

    def visitor_token(self):
        if self.interaction_store is None:
            return None
        if getattr(self, "_visitor", None):
            return self._visitor
        cookies = SimpleCookie()
        try:
            cookies.load(self.headers.get("Cookie", ""))
        except CookieError:
            pass
        token = cookies["datn_visitor"].value if "datn_visitor" in cookies else None
        if not self.interaction_store.exists(token):
            token = self.interaction_store.create_visitor()
            self._new_profile_cookie = token
        self._visitor = token
        return token

    def profile_cookie_header(self):
        token = getattr(self, "_new_profile_cookie", None)
        if token:
            self.send_header("Set-Cookie", f"datn_visitor={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=31536000")

    def log_message(self, fmt, *args):
        return

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.profile_cookie_header()
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/reviews":
            query = parse_qs(parsed.query)
            item = query.get("item_id", [""])[0]
            if item not in self.by_id:
                self.send_json({"error": "Unknown handset"}, 400)
                return
            rows = self.review_evidence.search(item, query.get("q", [""])[0]) if self.review_evidence else []
            self.send_json({"item_id": item, "reviews": rows, "notice": "Ý kiến công khai trong dữ liệu train; chưa xác minh, không dùng làm thông số hoặc lịch sử của bạn."})
            return
        if parsed.path == "/":
            self.visitor_token()
            data = HTML.replace('data-history="0"', 'data-history="1"' if self.interaction_store else 'data-history="0"')
            data = data.replace('__API_MODE__', 'Đã bật API theo tùy chọn' if self.llm_allowed else 'Không sử dụng API').encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.profile_cookie_header()
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/profile":
            if self.interaction_store is None:
                self.send_json({"error": "Local history is not enabled"}, 503)
            else:
                self.send_json({"profile": self.interaction_store.profile(self.visitor_token())})
            return
        if parsed.path == "/image":
            query = parse_qs(parsed.query)
            raw_path = unquote((query.get("path") or [""])[0])
            try:
                image_path = asset_path(raw_path).resolve()
                allowed_images = {asset_path(value).resolve() for doc in self.documents
                    for key in ("product_image_paths", "review_image_paths") for value in doc.get(key, [])}
            except (OSError, ValueError):
                self.send_error(404)
                return
            if image_path not in allowed_images or not image_path.is_file():
                self.send_error(404)
                return
            try:
                content_type = image_mime(image_path)
            except (OSError, ValueError):
                self.send_error(404)
                return
            data = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self):
        endpoint = urlparse(self.path).path
        if endpoint not in ("/api/query", "/api/recommend", "/api/events", "/api/profile/reset"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 65536:
                raise ValueError("Request too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_json({"error": "Invalid JSON"}, 400)
            return
        if not isinstance(payload, dict):
            self.send_json({"error": "Expected a JSON object"}, 400)
            return
        if endpoint in ("/api/events", "/api/profile/reset"):
            if self.interaction_store is None:
                self.send_json({"error": "Local history is not enabled"}, 503)
                return
            visitor = self.visitor_token()
            try:
                if endpoint == "/api/profile/reset":
                    self.interaction_store.reset(visitor)
                    self.sessions.reset_visitor(visitor)
                else:
                    if payload.get("kind") not in ("view", "like", "unlike", "save", "unsave"):
                        raise ValueError("Unknown product interaction")
                    self.interaction_store.record(visitor, payload.get("kind"),
                        item_id=payload.get("item_id"), event_id=payload.get("event_id"))
                self.send_json({"profile": self.interaction_store.profile(visitor)})
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        if endpoint == "/api/recommend":
            if self.local_preferences is not None:
                try:
                    if payload.get("user_index") is not None:
                        raise ValueError("Local visitor history cannot be combined with an offline demo user ID")
                    result = self.local_preferences.recommend(self.visitor_token(), payload.get("top_k", 5))
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, 400)
                    return
                self.send_json(result)
                return
            if self.recommender is None:
                self.send_json({"error": "Start with --enable-recommendations to load a trained recommender"}, 503)
                return
            try:
                result = self.recommender.recommend(payload.get("user_index"), payload.get("top_k", 5))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            self.send_json(result)
            return
        query = str(payload.get("query") or "").strip()
        if len(query) > 2000:
            self.send_json({"error": "Query must be at most 2000 characters"}, 400)
            return
        try:
            top_k = int(payload.get("top_k", 5))
            if not 1 <= top_k <= 100:
                raise ValueError("top_k out of range")
        except (TypeError, ValueError):
            self.send_json({"error": "top_k must be between 1 and 100"}, 400)
            return
        if not query:
            self.send_json({"query": query, "results": []})
            return
        visitor, preference_feed = None, None
        if self.interaction_store is not None:
            if payload.get("user_index") is not None:
                self.send_json({"error": "Local visitor history cannot use an offline demo user ID"}, 400)
                return
            visitor = self.visitor_token()
            self.interaction_store.record(visitor, "query", query=query)
            preference_feed = self.local_preferences.recommend(visitor, 100)
        if self.co_guiding is not None:
            try:
                with self.sessions.transaction(payload.get("session_id"), payload.get("user_index"), visitor_id=visitor) as (session_id, session):
                    result = self.co_guiding.run(query, user_index=session.user_index,
                        previous_plan=session.plan, mode=self.coupling_mode, top_k=top_k,
                        preference_feed=preference_feed,
                        seen_item_ids=preference_feed["profile"]["item_weights"] if preference_feed else None)
                    history = [] if result["query_plan"]["reset"] else session.turns
                    if result["abstention"]:
                        llm = {"answer": "Chưa có đủ dữ kiện đã xác minh để tìm sản phẩm đáp ứng các điều kiện này.", "enabled": False}
                    else:
                        llm = answer_policy(query, result["results"], allow_llm=self.llm_allowed,
                            requested=bool(payload.get("use_llm", True)), catalog_enabled=self.catalog_answers,
                            history=history, plan=result["query_plan"])
                    session.plan = result["query_plan"]
                    session.turns = (history + [{"query": query, "answer": llm.get("answer")}])[-8:]
                    self.send_json({**result, "session_id": session_id, "turns_retained": len(session.turns),
                        "answer": llm.get("answer"), "llm_error": llm.get("error"),
                        "llm_enabled": llm.get("enabled"), "llm_model": llm.get("model"),
                        "answer_mode": llm.get("mode", "abstention"), "llm_attempted": llm.get("attempted", False),
                        "profile_actions": preference_feed["profile"]["active_actions"] if preference_feed else {}})
                return
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            except (RuntimeError, OSError) as exc:
                self.send_json({"error": str(exc)}, 503)
                return
        use_query_llm = self.llm_allowed and bool(payload.get("use_query_llm", False))
        try:
            results = retrieve(self.index, self.documents, query, method=self.method,
                               dense=self.dense, top_k=top_k, use_llm=use_query_llm)
        except (ValueError, RuntimeError, OSError) as exc:
            self.send_json({"error": str(exc)}, 503)
            return
        query_plan = results[0].get("query_plan") if results else None
        use_llm = bool(payload.get("use_llm", True))
        llm = answer_policy(query, results, allow_llm=self.llm_allowed, requested=use_llm,
            catalog_enabled=self.catalog_answers, plan=query_plan)
        self.send_json({
            "query": query,
            "retrieval_method": self.method,
            "query_plan": query_plan,
            "results": results,
            "answer": llm.get("answer"),
            "llm_enabled": llm.get("enabled"),
            "llm_model": llm.get("model"),
            "answer_mode": llm.get("mode"), "llm_attempted": llm.get("attempted", False),
            "llm_error": llm.get("error"),
            "recommendations": preference_feed["results"][:top_k] if preference_feed else [],
            "profile_actions": preference_feed["profile"]["active_actions"] if preference_feed else {},
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--corpus", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "documents.jsonl")
    parser.add_argument("--index", type=Path, default=ROOT / 'artifacts/runtime/catalog' / "bm25_index.json")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--connected", action="store_true", help="Use the connected 4,435-item catalog and verified v2 recommender")
    parser.add_argument("--handsets", action="store_true", help="Use the new handset-only dataset, features and trained models")
    api_mode = parser.add_mutually_exclusive_group()
    api_mode.add_argument("--offline", action="store_true", help="Disable external generation/query APIs and use catalog summaries")
    api_mode.add_argument("--enable-llm", action="store_true", help="Explicitly enable optional generation; handset mode is offline by default")
    parser.add_argument("--historical-mixed-catalog", action="store_true", help="Reproduce earlier mixed phone/accessory demos; default scope is handsets only")
    parser.add_argument("--conversational-joint", choices=("gradient_decoupled", "joint_retrieval", "joint_bidirectional"),
                        help="Use a completed query-supervised model; requires --connected")
    parser.add_argument("--dense-index", type=Path, default=ROOT / 'artifacts/runtime/catalog/dense_index')
    parser.add_argument("--enable-recommendations", action="store_true")
    parser.add_argument("--recsys-method", default="graph_multimodal_gcl")
    parser.add_argument("--recsys-seed", type=int, default=42)
    parser.add_argument("--co-guiding", action="store_true")
    parser.add_argument("--coupling-mode", choices=("decoupled", "rag_to_rec", "rec_to_rag", "coupled"), default="coupled")
    parser.add_argument("--joint-ranking", action="store_true", help="Use the metadata-supervised joint ranking prototype")
    parser.add_argument("--interaction-db", type=Path, help="Persist this browser's actions in a local SQLite demo profile")
    args = parser.parse_args()
    ChatServer.llm_allowed = args.enable_llm
    ChatServer.catalog_answers = args.handsets or not args.enable_llm
    if args.handsets and (args.connected or args.historical_mixed_catalog or args.joint_ranking or args.conversational_joint):
        parser.error("--handsets uses the new handset-only models; do not combine with historical mixed-catalog flags or joint heads")
    if args.conversational_joint and not args.connected:
        parser.error("--conversational-joint requires --connected")
    if args.connected and args.joint_ranking:
        parser.error("--connected cannot use the old metadata-proxy --joint-ranking")
    args.method = args.method or ("multimodal" if args.connected or args.handsets else "legacy")
    connected = None
    if args.connected or args.handsets:
        if args.handsets:
            from phone_assistant.retrieval.serving.handset_serving import HandsetRuntime
            connected = HandsetRuntime()
        else:
            from phone_assistant.retrieval.serving.connected_serving import ConnectedRuntime
            connected = ConnectedRuntime()
        args.corpus, args.index, args.dense_index = connected.corpus, connected.index, connected.dense
        args.co_guiding = True
    if args.joint_ranking and not args.co_guiding:
        parser.error("--joint-ranking requires --co-guiding")

    ChatServer.documents = read_jsonl(args.corpus.resolve())
    all_documents = ChatServer.documents
    if not args.historical_mixed_catalog:
        from phone_assistant.data.phone_scope import handset_documents
        ChatServer.documents = handset_documents(all_documents)
        if not ChatServer.documents:
            parser.error("No verified handset candidates in this corpus")
    ChatServer.by_id = {doc["doc_id"]: doc for doc in ChatServer.documents}
    if args.handsets:
        from phone_assistant.conversation.handset_reviews import ReviewEvidence, STORE
        ChatServer.review_evidence = ReviewEvidence() if (STORE / "manifest.json").exists() else None
    ChatServer.index = json.loads(args.index.resolve().read_text(encoding="utf-8"))
    if not args.historical_mixed_catalog:
        from phone_assistant.retrieval.search.bm25 import build_index
        ChatServer.index = build_index(ChatServer.documents)
    ChatServer.method = args.method
    if args.interaction_db:
        from phone_assistant.conversation.interaction_store import InteractionStore, LocalPreferenceService
        ChatServer.interaction_store = InteractionStore(args.interaction_db, ChatServer.by_id)
        ChatServer.local_preferences = LocalPreferenceService(ChatServer.documents, ChatServer.interaction_store)
    if args.enable_recommendations or args.co_guiding:
        from phone_assistant.recommendation.recsys_service import RecommendationService
        if connected is not None:
            ChatServer.recommender = connected.recommender(args.recsys_method, args.recsys_seed, args.conversational_joint)
        elif args.joint_ranking:
            ChatServer.recommender = RecommendationService(models=ROOT/'artifacts/models/joint',
                experiment=ROOT/'artifacts/evaluation/results_joint_ranking/experiment.json',method='joint_bidirectional',seed=args.recsys_seed)
        else:
            ChatServer.recommender = RecommendationService(method=args.recsys_method, seed=args.recsys_seed, corpus=args.corpus)
        if not args.historical_mixed_catalog:
            from phone_assistant.data.phone_scope import HandsetRecommender
            ChatServer.recommender = HandsetRecommender(ChatServer.recommender, ChatServer.documents)
    if args.method != "legacy":
        from phone_assistant.retrieval.search.dense_search import DenseRetriever, text_document
        from phone_assistant.retrieval.search.bm25 import build_index
        if connected is not None:
            from phone_assistant.retrieval.serving.connected_serving import ConnectedDenseRetriever
            ChatServer.dense = ConnectedDenseRetriever(args.dense_index, all_documents)
        else:
            ChatServer.dense = DenseRetriever(args.dense_index, all_documents)
        if not args.historical_mixed_catalog:
            from phone_assistant.data.phone_scope import HandsetRanking
            ChatServer.dense = HandsetRanking(ChatServer.dense, ChatServer.by_id, len(all_documents))
        ChatServer.index = build_index([
            {**doc, "text": text_document(doc)} for doc in ChatServer.documents
        ])
    if args.co_guiding:
        from phone_assistant.conversation.co_guiding import CoGuidingEngine
        joint_query=None
        if args.joint_ranking:
            from phone_assistant.recommendation.joint_ranking import JointQueryRanking
            joint_query=JointQueryRanking(ChatServer.recommender,ROOT/'artifacts/models/joint',ROOT/'artifacts/evaluation/results_joint_ranking/experiment.json')
        kwargs = {}
        if connected is not None:
            from phone_assistant.retrieval.queries.product_constraints import ProductQueryPlanner, connected_facts, connected_verdict
            kwargs = dict(planner=ProductQueryPlanner(ChatServer.documents), fact_extractor=connected_facts, verdict=connected_verdict)
            if args.handsets:
                from phone_assistant.conversation.handset_dialogue import dialogue_options
                kwargs = dialogue_options(connected, ChatServer.documents)
            if args.conversational_joint:
                from phone_assistant.retrieval.serving.connected_serving import ConversationalQueryRanking
                joint_query = ConversationalQueryRanking(ChatServer.recommender, connected.joint_models,
                    connected.joint_experiment, dense=ChatServer.dense)
        if joint_query is not None and not args.historical_mixed_catalog:
            from phone_assistant.data.phone_scope import HandsetRanking
            joint_query = HandsetRanking(joint_query, ChatServer.by_id, len(all_documents))
        ChatServer.co_guiding = CoGuidingEngine(ChatServer.documents, ChatServer.index,
            ChatServer.recommender, dense=ChatServer.dense, method=args.method,joint_query=joint_query, **kwargs)
        ChatServer.coupling_mode = args.coupling_mode

    server = ThreadingHTTPServer((args.host, args.port), ChatServer)
    print(f"Chat UI: http://{args.host}:{args.port}")
    print(f"Retrieval method: {args.method}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
