# python parser

import ply
import ply.yacc
import shll_lex
from shll_ast import *

class Parser:
    tokens = shll_lex.Lexer.tokens

    start = 'program'

    precedence = (
        ('left', 'ELSE'),
        ('left', 'IN'),
        ('left', '|'), # boolean operators
        ('left', 'EMPTY_TYPE_PARAM_LIST'),
        ('left', '<', '>'), # integer comparison operators
        ('left', '+'), # integer math operators
        ('left', '.'),
        )

    def p_program(self, p):
        'program : '
        p[0] = Program()

    def p_program_type(self, p):
        'program : program strdef'
        p[0] = p[1].add_type(p[2])

    def p_program_task(self, p):
        'program : program taskdef'
        p[0] = p[1].add_task(p[2])

    def p_strdef(self, p):
        'strdef : TYPE ID opt_type_params "=" type'
        p[0] = Typedef(name = p[2], params = p[3], innertype = p[5])

    def p_inttype(self, p):
        'type : INT'
        p[0] = IntType()

    def p_booltype(self, p):
        'type : BOOL'
        p[0] = BoolType()

    def p_ptrtype_single(self, p):
        'type : type "@" ID'
        p[0] = PtrType(elemtype = p[1], regions = [ p[3] ])

    def p_ptrtype_union(self, p):
        'type : type "@" "(" region_list ")"'
        p[0] = PtrType(elemtype = p[1], regions = p[4])

    def p_tupletype(self, p):
        'type : "<" type "," type ">"'
        p[0] = TupleType(lhs = p[2], rhs = p[4])

    def p_usertype(self, p):
        'type : ID opt_type_params'
        p[0] = UserType(name = p[1], params = p[2])

    def p_opt_type_params(self, p):
        '''opt_type_params : %prec EMPTY_TYPE_PARAM_LIST \n| "<" region_list ">"'''
        if len(p) > 1:
            p[0] = p[2]
        else:
            p[0] = []

    def p_opt_task_params(self, p):
        '''opt_task_params : \n| "[" region_list "]"'''
        if len(p) > 1:
            p[0] = p[2]
        else:
            p[0] = []

    def p_rrtype(self, p):
        'type : RR "[" region_list "]" type WHERE const_list'
        p[0] = RRType(regions = p[3],
                      innertype = p[5],
                      constraints = p[7])

    def p_region_list(self, p):
        '''region_list : ID \n | region_list "," ID'''
        if len(p) == 2:
            p[0] = [ p[1] ]
        else:
            p[0] = p[1] + [ p[3] ]

    def p_const_list(self, p):
        '''const_list : region_const \n | const_list AND region_const'''
        if len(p) == 2:
            p[0] = [ p[1] ]
        else:
            p[0] = p[1] + [ p[3] ]

    def p_region_const(self, p):
        '''region_const : ID "*" ID \n | ID SUBSET ID'''
        p[0] = RegionConstraint(op = p[2], lhs = p[1], rhs = p[3])

    # def p_const_subregion(self, p):
    #     'const_list : ID SUBSET ID \n | const_list AND ID SUBSET ID'
    #     if len(p) == 4:
    #         p[0] = RegionConstraints().add_subregion(p[1], p[3])
    #     else:
    #         p[0] = p[1].add_subregion(p[3], p[5])

    # def p_const_disjoint(self, p):
    #     'const_list : ID "*" ID \n | const_list AND ID "*" ID'
    #     if len(p) == 4:
    #         p[0] = RegionConstraints().add_disjoint(p[1], p[3])
    #     else:
    #         p[0] = p[1].add_disjoint(p[3], p[5])

    def p_taskdef(self, p):
        'taskdef : TASK ID opt_task_params "(" formal_list ")" ":" type effects "=" expr'
        p[0] = Taskdef(name = p[2],
                       params = p[3],
                       formals = p[5],
                       rettype = p[8],
                       effects = p[9],
                       body = p[11])

    def p_formal_list(self, p):
        '''formal_list : ID ":" type
                       | formal_list "," ID ":" type'''
        if len(p) == 4:
            p[0] = FormalsList()
            p[0].add_formal(fname = p[1], ftype = p[3])
        else:
            p[0] = p[1]
            p[0].add_formal(fname = p[3], ftype = p[5])

    def p_effects(self, p):
        'effects : '
        p[0] = Effects()

    def p_effects_reads(self, p):
        'effects : effects "," READS "(" region_list ")"'
        p[0] = p[1].add_reads(p[5])

    def p_effects_writes(self, p):
        'effects : effects "," WRITES "(" region_list ")"'
        p[0] = p[1].add_writes(p[5])

    def p_effects_rdwrs(self, p):
        'effects : effects "," RDWRS "(" region_list ")"'
        p[0] = p[1].add_reads(p[5]).add_writes(p[5])

    def p_effects_reduces(self, p):
        'effects : effects "," REDUCES "(" ID "," region_list ")"'
        p[0] = p[1].add_reduces(p[5], p[7])

    def p_letexpr(self, p):
        'expr : LET ID ":" type "=" expr IN expr'
        p[0] = LetExpr(valname = p[2], valtype = p[4], valexpr = p[6],
                       body = p[8])
        
    def p_identexpr(self, p):
        'expr : ID'
        p[0] = IdentExpr(p[1])

    def p_tupleexpr(self, p):
        'expr : "<" expr "," expr ">"'
        p[0] = TupleExpr(lhs = p[2], rhs = p[4])

    def p_readexpr(self, p):
        'expr : READ "(" expr ")"'
        p[0] = ReadExpr(ptrexpr = p[3])

    def p_writeexpr(self, p):
        'expr : WRITE "(" expr "," expr ")"'
        p[0] = WriteExpr(ptrexpr = p[3], valexpr = p[5])

    def p_reduceexpr(self, p):
        'expr : REDUCE "(" ID "," expr "," expr ")"'
        p[0] = ReduceExpr(func = p[3], ptrexpr = p[5], valexpr = p[7])

    def p_fieldexpr(self, p):
        'expr : expr "." INTVAL'
        p[0] = FieldExpr(subexpr = p[1], field = p[3])

    def p_binopexpr(self, p):
        '''expr : expr "<" expr
                | expr ">" expr
                | expr "+" expr
                | expr "|" expr'''
        p[0] = BinOpExpr(op = p[2], lhs = p[1], rhs = p[3])

    def p_ifexpr(self, p):
        'expr : IF expr THEN expr ELSE expr'
        p[0] = IfExpr(condexpr = p[2], thenexpr = p[4], elseexpr = p[6])

    def p_callexpr(self, p):
        'expr : ID opt_task_params "(" arg_list ")"'
        p[0] = CallExpr(name = p[1],
                        params = p[2],
                        args = p[4])

    def p_arg_list(self, p):
        '''arg_list : expr \n | arg_list "," expr'''
        if len(p) == 2:
            p[0] = [ p[1] ]
        else:
            p[0] = p[1] + [ p[3] ]

    def p_newexpr(self, p):
        'expr : NEW type'
        p[0] = NewExpr(p[2])

    def p_partexpr(self, p):
        'expr : PARTITION ID USING ID opt_task_params "(" arg_list ")" AS region_list IN expr'
        p[0] = PartitionExpr(region = p[2],
                             cf_name = p[4],
                             cf_params = p[5],
                             cf_args = p[7],
                             subregions = p[10],
                             body = p[12])

    def p_packexpr(self, p):
        'expr : PACK expr AS type "[" region_list "]"'
        p[0] = PackExpr(body = p[2],
                        rrtype = p[4],
                        rrparams = p[6])

    def p_unpackexpr(self, p):
        'expr : UNPACK expr AS ID ":" type "[" region_list "]" IN expr'
        p[0] = UnpackExpr(argexpr = p[2],
                          name = p[4],
                          rrtype = p[6],
                          rrparams = p[8],
                          body = p[11])

    def p_uprgnexpr(self, p):
        'expr : UPREGION "(" expr "," region_list ")"'
        p[0] = UpRegionExpr(ptrexpr = p[3],
                            regions = p[5])

    def p_dnrgnexpr(self, p):
        'expr : DOWNREGION "(" expr "," region_list ")"'
        p[0] = DownRegionExpr(ptrexpr = p[3],
                              regions = p[5])

    def p_parenexpr(self, p):
        'expr : "(" expr ")"'
        p[0] = p[2]

    def p_isnullexpr(self, p):
        'expr : ISNULL "(" expr ")"'
        p[0] = IsNullExpr(p[3])

    def p_nullconst(self, p):
        'expr : NULL'
        p[0] = NullConstExpr()

    def p_intconst(self, p):
        'expr : INTVAL'
        p[0] = IntConstExpr(value = p[1])

    def p_boolconst(self, p):
        '''expr : TRUE \n | FALSE'''
        p[0] = BoolConstExpr(value = (p[1] == 'true'))

    def __init__(self, lexer=None, **kwargs):
        if lexer is None:
            lexer = shll_lex.Lexer()
        self.lexer = lexer
        self.parser = ply.yacc.yacc(module = self, **kwargs)

    def parse(self, src, **kwargs):
        self.lexer.input(src)
        return self.parser.parse(lexer = self.lexer)

if __name__ == '__main__':
    import sys
    p = Parser()
    pgrm = p.parse(sys.stdin)
    pgrm.type_check()
