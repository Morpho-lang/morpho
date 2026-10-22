"""Lightweight documentation highlighting based on src/support/lex.c."""

from pygments.lexer import RegexLexer, include, words
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Text


class MorphoLexer(RegexLexer):
    name = 'Morpho'
    aliases = ['morpho']
    filenames = ['*.morpho']

    tokens = {
        'root': [
            (r'\A#![^\n]*', Comment.Hashbang),
            (r'\s+', Text.Whitespace),
            (r'//[^\n]*', Comment.Single),
            (r'/\*', Comment.Multiline, 'comment'),
            (r'"', String.Double, 'string'),
            (words(('true', 'false', 'nil'), suffix=r'\b'), Keyword.Constant),
            (words(('and', 'or'), suffix=r'\b'), Operator.Word),
            (words(('as', 'break', 'class', 'continue', 'catch', 'do', 'else',
                    'for', 'fn', 'help', 'if', 'in', 'is', 'import', 'im',
                    'print', 'return', 'self', 'super', 'try', 'var', 'while',
                    'with'), suffix=r'\b'), Keyword),
            (r'\d+(?:\.(?=\d|\s|$)\d*)?(?:[eE][+-]?\d+)?im\b', Number),
            (r'\d+\.(?=\d|\s|$)\d*(?:[eE][+-]?\d+)?', Number.Float),
            (r'\d+[eE][+-]?\d+', Number.Float),
            (r'\d+', Number.Integer),
            (r'[A-Za-z_][A-Za-z_0-9]*', Name),
            (r'\.\.\.|\.\.|[+*/=!<>-]=|&&|\|\||[\^?@#.+=!<>&|*/-]', Operator),
            (r'[()\[\]{};:,]', Punctuation),
        ],
        'comment': [
            (r'/\*', Comment.Multiline, '#push'),
            (r'\*/', Comment.Multiline, '#pop'),
            (r'[^*/]+|[*/]', Comment.Multiline),
        ],
        'string': [
            (r'\\[\s\S]', String.Escape),
            (r'\$\{', String.Interpol, 'interpolation'),
            (r'"', String.Double, '#pop'),
            (r'[^"\\$]+|\$', String.Double),
        ],
        'interpolation': [
            # Morpho resumes string scanning at the next closing brace.
            (r'\}', String.Interpol, '#pop'),
            include('root'),
        ],
    }


def setup(app):
    app.add_lexer('morpho', MorphoLexer)
    return {'version': '1.0', 'parallel_read_safe': True,
            'parallel_write_safe': True}
