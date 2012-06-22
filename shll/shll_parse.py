# python parser

import ply
import ply.yacc
import shll_lex

class Parser:
    tokens = shll_lex.Lexer.tokens

    start = 'program'

    precedence = (
        ('left', 'ELSE'),
        ('left', 'IN'),
        ('left', '.'),
        ('left', '+'), # integer math operators
        ('left', 'LT', 'GT'), # integer comparison operators
        ('left', '|'), # boolean operators
        )

    def p_program(self, p):
        '''program : 
                | program strdef
                | program taskdef'''
        if len(p) == 1:
            p[0] = []
        else:
            p[0] = p[1] + list(p[2])

    def p_strdef(self, p):
        'strdef : TYPE ID opt_region_params "=" type'
        p[0] = dict(name = p[2], params = p[3], innertype = p[5])

    def p_type(self, p):
        '''type : basetype
                | ptrtype
                | tupletype
                | usertype
                | rrtype'''
        p[0] = p[1]

    def p_basetype(self, p):
        '''basetype : INT \n | BOOL'''
        p[0] = dict(name = p[1], params = [])

    def p_ptrtype(self, p):
        'ptrtype : type "@" ID'
        p[0] = dict(elemtype = p[1], region = p[3])

    def p_tupletype(self, p):
        'tupletype : "<" type "," type ">"'
        p[0] = dict(lhs = p[2], rhs = p[4])

    def p_usertype(self, p):
        'usertype : ID opt_region_params'
        p[0] = dict(name = p[1], params = p[2])

    def p_opt_region_params(self, p):
        '''opt_region_params : \n | "<" region_list ">"'''
        if len(p) > 1:
            p[0] = p[2]
        else:
            p[0] = []

    def p_rrtype(self, p):
        'rrtype : RR "(" region_list ")" type WHERE const_list'
        p[0] = dict(regions = p[3],
                    innertype = p[5],
                    constraints = p[7])

    def p_region_list(self, p):
        '''region_list : ID \n | region_list "," ID'''
        if len(p) == 2:
            p[0] = [ p[1] ]
        else:
            p[0] = p[1] + list(p[2])

    def p_const_list(self, p):
        '''const_list : region_const \n | const_list AND region_const'''
        if len(p) == 2:
            p[0] = [ p[1] ]
        else:
            p[0] = p[1] + list(p[2])

    def p_region_const(self, p):
        '''region_const : ID "*" ID \n | ID SUBSET ID'''
        p[0] = dict(op = p[2], lhs = p[1], rhs = p[3])
        
    def p_taskdef(self, p):
        'taskdef : TASK ID opt_region_params "(" formal_list ")" ":" type effects "=" expr'
        p[0] = dict(name = p[2],
                    params = p[3],
                    formals = p[5],
                    rettype = p[8],
                    effects = p[9],
                    body = p[11])

    def p_formal_list(self, p):
        '''formal_list : ID ":" type
                       | formal_list "," ID ":" type'''
        if len(p) == 4:
            p[0] = dict(order = [ p[1] ],
                        byname = { p[1]: p[3] })
        else:
            p[0] = p[1]
            p[0]['order'].append(p[2])
            p[0]['byname'][p[2]] = p[4]

    def p_effects(self, p):
        '''effects :
                   | effects "," READS '(' region_list ')'
                   | effects "," RDWRS '(' region_list ')'
                   | effects "," WRITES '(' region_list ')'
                   | effects "," REDUCES '(' ID "," region_list ')' '''
        if len(p) == 1:
            p[0] = dict(reads = set(), writes = set(), reduces = dict())
        else:
            #for i, x in enumerate(p): print i, x
            p[0] = p[1]
            if (p[3] == 'reads') or (p[3] == 'rdwrs'):
                p[0]['reads'] |= set(p[5])
            if (p[3] == 'writes') or (p[3] == 'rdwrs'):
                p[0]['writes'] |= set(p[5])
            if (p[3] == 'reduces'):
                if p[5] not in p[0]['reduces']:
                    p[0]['reduces'][p[5]] = set()
                p[0]['reduces'][p[5]] |= set(p[7])

    def p_expr(self, p):
        '''expr : letexpr 
                | identexpr
                | tupleexpr
                | readexpr
                | writeexpr
                | reduceexpr
                | fieldexpr
                | binopexpr
                | ifexpr
                | callexpr
                | newexpr
                | partexpr
                | packexpr
                | unpackexpr
                | uprgnexpr
                | dnrgnexpr
                | parenexpr
                | isnullexpr
                | nullconst
                | intconst
                | boolconst'''
        p[0] = p[1]

    def p_letexpr(self, p):
        'letexpr : LET ID ":" type "=" expr IN expr'
        p[0] = dict(varname = p[1], vartype = p[3], value = p[5], body = p[7])
        
    def p_identexpr(self, p):
        'identexpr : ID'
        p[0] = p[1]

    def p_tupleexpr(self, p):
        'tupleexpr : "<" expr "," expr ">"'
        p[0] = dict(lhs = p[1], rhs = p[3])

    def p_readexpr(self, p):
        'readexpr : READ "(" expr ")"'
        p[0] = dict(ptr = p[3])

    def p_writeexpr(self, p):
        'writeexpr : WRITE "(" expr "," expr ")"'
        p[0] = dict(ptr = p[3], val = p[5])

    def p_reduceexpr(self, p):
        'reduceexpr : REDUCE "(" ID "," expr "," expr ")"'
        p[0] = dict(func = p[3], ptr = p[5], val = p[7])

    def p_fieldexpr(self, p):
        'fieldexpr : expr "." INTVAL'
        p[0] = dict(subexpr = p[1], field = p[3])

    def p_binopexpr(self, p):
        '''binopexpr : expr LT expr
                     | expr GT expr
                     | expr "+" expr
                     | expr "|" expr'''
        p[0] = dict(op = p[2], lhs = p[1], rhs = p[3])

    def p_ifexpr(self, p):
        'ifexpr : IF expr THEN expr ELSE expr'
        p[0] = dict(condexpr = p[2], thenexpr = p[4], elseexpr = p[6])

    def p_callexpr(self, p):
        'callexpr : ID opt_region_params "(" arg_list ")"'
        p[0] = dict(name = p[1],
                    params = p[2],
                    args = p[4])

    def p_arg_list(self, p):
        '''arg_list : expr \n | arg_list "," expr'''
        if len(p) == 2:
            p[0] = [ p[1] ]
        else:
            p[0] = p[1] + list(p[2])

    def p_newexpr(self, p):
        'newexpr : NEW ptrtype'
        p[0] = p[2]

    def p_partexpr(self, p):
        'partexpr : PARTITION ID USING ID opt_region_params "(" arg_list ")" AS region_list IN expr'
        p[0] = dict(r_in = p[2],
                    cf_name = p[4],
                    cf_params = p[5],
                    cf_args = p[7],
                    subregion_names = p[9],
                    body = p[11])

    def p_packexpr(self, p):
        'packexpr : PACK expr AS type "(" region_list ")"'
        p[0] = dict(body = p[2],
                    rrtype = p[4],
                    rrparams = p[6])

    def p_unpackexpr(self, p):
        'unpackexpr : UNPACK expr AS ID ":" type "(" region_list ")" IN expr'
        p[0] = dict(v_in = p[2],
                    name = p[4],
                    rrtype = p[6],
                    rrparams = p[8],
                    body = p[10])

    def p_uprgnexpr(self, p):
        'uprgnexpr : UPREGION "(" expr "," ID ")"'
        p[0] = dict(expr = p[3],
                    region = p[5])

    def p_dnrgnexpr(self, p):
        'dnrgnexpr : DOWNREGION "(" expr "," ID ")"'
        p[0] = dict(expr = p[3],
                    region = p[5])

    def p_parenexpr(self, p):
        'parenexpr : "(" expr ")"'
        p[0] = p[2]

    def p_isnullexpr(self, p):
        'isnullexpr : ISNULL "(" expr ")"'
        p[0] = p[3]

    def p_nullconst(self, p):
        'nullconst : NULL'
        p[0] = dict(value = p[1])

    def p_intconst(self, p):
        'intconst : INTVAL'
        p[0] = dict(value = p[1])

    def p_boolconst(self, p):
        '''boolconst : TRUE \n | FALSE'''
        p[0] = dict(value = (p[1] == 'true'))

    def __init__(self, lexer=None, **kwargs):
        if lexer is None:
            lexer = shll_lex.Lexer()
        self.lexer = lexer
        self.parser = ply.yacc.yacc(module = self, **kwargs)

    def parse(self, src, **kwargs):
        self.lexer.input(src)
        self.parser.parse(lexer = self.lexer)

if __name__ == '__main__':
    import sys
    p = Parser()
    p.parse(sys.stdin)
