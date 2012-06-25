# python lexer

import ply
from ply.lex import TOKEN

class Lexer:
    keywords = ('task', 'let', 'in', 'type', 'int', 'bool', 'where', 'rr',
                'and', 'reads', 'writes', 'rdwrs', 'reduces', 'true', 'false',
                'read', 'write', 'reduce', 'if', 'then', 'else', 'new',
                'isnull', 'null', 'partition', 'using', 'as', 'pack', 'unpack',
                'upregion', 'downregion')

    tokens = [
        'ID',
        'LT',
        'GT',
        'SUBSET',
        'INTVAL',
        ] + list(x.upper() for x in keywords)

    literals = '={}()<>@*,:.+&|[]'

    whitespace = " \t\r"

    identifier = r'([A-Za-z][A-Za-z0-9_]*)'

    t_ignore = whitespace

    @TOKEN(r'/\*([^*]+|(\*[^\/]))*\*/')
    def t_ignore_comment(self, t):
        pass

    def t_error(self, t):
        raise ply.lex.LexError("Illegal character on line " + str(t.lexer.lineno) + ": " + t.value, t.value)

    @TOKEN(r'\n+')
    def t_ignore_newlines(self, t):
        t.lexer.lineno += len(t.value)

    @TOKEN(identifier)
    def t_ID(self, t):
        if t.value in self.keywords:
            t.type = t.value.upper()
        return t

    @TOKEN(r'<=')
    def t_SUBSET(self, t):
        return t

    @TOKEN(r'\.lt\.')
    def t_LT(self, t):
        return t

    @TOKEN(r'\.gt\.')
    def t_GT(self, t):
        return t

    @TOKEN(r'[0-9]+')
    def t_INTVAL(self, t):
        t.value = int(t.value)
        return t

    def __init__(self, src=None, **kwargs):
        self.lexer = ply.lex.lex(module = self, **kwargs)
        if src is not None:
            self.input(src)

    def input(self, src):
        self.lexer.input(src.read())

    def token(self, **kwargs):
        return self.lexer.token(**kwargs)

if __name__ == '__main__':
    import sys
    l = Lexer(sys.stdin)
    for tok in l.lexer:
        print tok
