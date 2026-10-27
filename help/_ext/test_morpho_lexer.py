"""Run with: python -m unittest discover -s help/_ext -p 'test_*.py'."""

import unittest

from pygments.token import Comment, Error, Keyword, Number, Operator, String

from morpho_lexer import MorphoLexer


class MorphoLexerTests(unittest.TestCase):
    def tokens(self, source):
        tokens = list(MorphoLexer().get_tokens(source))
        self.assertFalse(any(token in Error for token, _ in tokens))
        return tokens

    def test_keywords_and_ranges(self):
        tokens = self.tokens('var x = true and nil; for (i in 1..4) print i')
        for expected in [(Keyword, 'var'), (Keyword.Constant, 'true'),
                         (Keyword.Constant, 'nil'), (Operator.Word, 'and'),
                         (Operator, '..'), (Number.Integer, '1')]:
            self.assertIn(expected, tokens)

    def test_numbers(self):
        tokens = self.tokens('12 1.25 2e-3 4im 1.2e+3im 1...4')
        for expected in [(Number.Integer, '12'), (Number.Float, '1.25'),
                         (Number.Float, '2e-3'), (Number, '4im'),
                         (Number, '1.2e+3im'), (Operator, '...')]:
            self.assertIn(expected, tokens)

    def test_nested_comments(self):
        tokens = self.tokens('/* outer /* inner */ outer */ var x // line')
        self.assertIn((Keyword, 'var'), tokens)
        self.assertIn((Comment.Single, '// line'), tokens)
        self.assertFalse(any(value.strip() == 'outer' and token not in Comment
                             for token, value in tokens))

    def test_strings_and_interpolation(self):
        tokens = self.tokens(r'"escaped \" and \${literal}; ${1 + 2}"')
        self.assertIn((String.Escape, '\\"'), tokens)
        self.assertIn((String.Escape, '\\$'), tokens)
        self.assertIn((String.Interpol, '${'), tokens)
        self.assertIn((String.Interpol, '}'), tokens)
        self.assertIn((Number.Integer, '2'), tokens)


if __name__ == '__main__':
    unittest.main()
