from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phone_assistant.reporting.chapters.export_chapters_docx import INLINE_TOKEN, normalize_markdown


class MarkdownDialectTests(unittest.TestCase):
    def test_multiline_display_math_becomes_one_line_and_keeps_the_tag(self):
        text = "Before\n\n$$\nX=\\left(u,q\\right).\n\\tag{1.1}\n$$\n\nAfter"
        self.assertIn(r"$$ X=\left(u,q\right). \tag{1.1} $$", normalize_markdown(text).splitlines())

    def test_inline_dollar_math_is_converted_but_display_and_escaped_dollars_are_left_alone(self):
        text = r"Gọi $\mathcal U$ và $i$ là hai tập." + "\n" + r"$$ a=b \tag{2.1} $$" + "\n" + r"Giá \$5 không phải công thức."
        result = normalize_markdown(text)
        self.assertIn(r"Gọi \(\mathcal U\) và \(i\) là hai tập.", result)
        self.assertIn(r"$$ a=b \tag{2.1} $$", result)
        self.assertIn(r"Giá \$5 không phải công thức.", result)

    def test_italic_bold_code_and_math_tokens_are_recognised_in_order(self):
        tokens = [m.group() for m in INLINE_TOKEN.finditer(r"**đậm** *nghiêng* `mã` \(x*y\) [a](http://b)")]
        self.assertEqual(tokens, ["**đậm**", "*nghiêng*", "`mã`", r"\(x*y\)", "[a](http://b)"])


if __name__ == "__main__":
    unittest.main()
