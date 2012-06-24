# AST nodes for python parser

class DuplicateTypeDefinition(Exception):
    pass

class DuplicateTaskDefinition(Exception):
    pass

class Program:
    def __init__(self):
        self.types = dict()
        self.tasks = dict()

    def add_type(self, t):
        if t.name in self.types:
            raise DuplicateTypeDefinition(t.name, self.types[t.name], t)
        print "adding type", t.name, "=", str(t.innertype)
        self.types[t.name] = t
        return self

    def add_task(self, t):
        if t.name in self.tasks:
            raise DuplicateTaskDefinition(t.name, self.tasks[t.name], t)
        print "adding task", str(t)
        self.tasks[t.name] = t
        return self

class Typedef:
    def __init__(self, name, params, innertype):
        self.name = name
        self.params = params
        self.innertype = innertype

class Taskdef:
    def __init__(self, name, params, formals, rettype, effects, body):
        self.name = name
        self.params = params
        self.formals = formals
        self.rettype = rettype
        self.effects = effects
        self.body = body

    def __str__(self):
        s = self.name
        if len(self.params) > 0:
            s = s + "<" + ", ".join(self.params) + ">"
        s += "("
        s += str(self.formals)
        s += ") : " + str(self.rettype)
        s += str(self.effects)
        return s

class Type:
    pass

class IntType(Type):
    def __init__(self):
        pass

    def __str__(self):
        return "int"

class BoolType(Type):
    def __init__(self):
        pass

    def __str__(self):
        return "bool"

class PtrType(Type):
    def __init__(self, elemtype, region):
        self.elemtype = elemtype
        self.region = region

    def __str__(self):
        return str(self.elemtype) + "@" + str(self.region)

class TupleType(Type):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return ("<%s, %s>" % (self.lhs, self.rhs))

class UserType(Type):
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def __str__(self):
        if len(self.params) > 0:
            return ("%s<%s>" % (self.name,
                                ", ".join(self.params)))
        else:
            return self.name

class RRType(Type):
    def __init__(self, regions, innertype, constraints):
        self.regions = regions
        self.innertype = innertype
        self.constraints = constraints

    def __str__(self):
        return ("rr(%s) %s where %s" % (", ".join(self.regions),
                                        self.innertype,
                                        " and ".join(map(str, self.constraints))))

class RegionConstraint:
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return ("%s %s %s" % (self.lhs, self.op, self.rhs))

class FormalsList:
    def __init__(self):
        self.byorder = []
        self.byname = {}

    def add_formal(self, fname, ftype):
        if fname in self.byname:
            raise DuplicateFormalParameter(self, fname)
        self.byorder.append(fname)
        self.byname[fname] = ftype

    def __str__(self):
        if len(self.byorder) > 0:
            return (", ".join(("%s: %s" % (x, self.byname[x])) for x in self.byorder))
        else:
            return ""

class Effects:
    def __init__(self):
        self.reads = set()
        self.writes = set()
        self.reduces = dict()

    def add_reads(self, regions):
        self.reads |= set(regions)
        return self

    def add_writes(self, regions):
        self.writes |= set(regions)
        return self

    def add_reduces(self, fname, regions):
        if fname not in self.reduces:
            self.reduces[fname] = set()
        self.reduces[fname] |= set(regions)
        return self

    def __str__(self):
        s = ""
        if len(self.reads) > 0:
            s += ", reads(" + ", ".join(self.reads) + ")"
        if len(self.writes) > 0:
            s += ", writes(" + ", ".join(self.writes) + ")"
        for fname, rset in self.reduces:
            s += ", reduces(" + fname + ", " + ", ".join(rset) + ")"
        return s

class Expr:
    def __init__(self):
        pass

class LetExpr(Expr):
    def __init__(self, valname, valtype, valexpr, body):
        self.valname = valname
        self.valtype = valtype
        self.valexpr = valexpr
        self.body = body

    def __str__(self):
        return ("let %s : %s = %s\n in %s" %
                (self.valname, self.valtype, self.valexpr,
                 self.body))

class IdentExpr(Expr):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

class TupleExpr(Expr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return ("<%s, %s>" % (self.lhs, self.rhs))

class FieldExpr(Expr):
    def __init__(self, subexpr, field):
        self.subexpr = subexpr
        self.field = field

    def __str__(self):
        return ("%s.%d" % (self.subexpr, self.field))

class ReadExpr(Expr):
    def __init__(self, ptrexpr):
        self.ptrexpr = ptrexpr

    def __str__(self):
        return ("read(%s)" % (self.ptrexpr,))

class WriteExpr(Expr):
    def __init__(self, ptrexpr, valexpr):
        self.ptrexpr = ptrexpr
        self.valexpr = valexpr

    def __str__(self):
        return ("write(%s, %s)" % (self.ptrexpr, self.valexpr))

class ReduceExpr(Expr):
    def __init__(self, func, ptrexpr, valexpr):
        self.func = func
        self.ptrexpr = ptrexpr
        self.valexpr = valexpr

    def __str__(self):
        return ("reduce(%s, %s, %s)" % (self.func, self.ptrexpr, self.valexpr))

class BinOpExpr(Expr):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return ("%s %s %s" % (self.lhs, self.op, self.rhs))

class IfExpr(Expr):
    def __init__(self, condexpr, thenexpr, elseexpr):
        self.condexpr = condexpr
        self.thenexpr = thenexpr
        self.elseexpr = elseexpr

    def __str__(self):
        return ("if %s then %s else %s" %
                (self.condexpr, self.thenexpr, self.elseexpr))

class CallExpr(Expr):
    def __init__(self, name, params, args):
        self.name = name
        self.params = params
        self.args = args

    def __str__(self):
        return ("%s%s(%s)" %
                (self.name,
                 ("<" + ", ".join(self.params) + ">" if len(self.params) > 0 else ""),
                 ", ".join(self.args)))

class NewExpr(Expr):
    def __init__(self, ptrtype):
        self.ptrtype = ptrtype

    def __str__(self):
        return ("new %s" % (self.ptrtype,))

class PartitionExpr(Expr):
    def __init__(self, region, cf_name, cf_params, cf_args, subregions, body):
        self.region = region
        self.cf_name = cf_name
        self.cf_params = cf_params
        self.cf_args = cf_args
        self.subregions = subregions
        self.body = body

    def __str__(self):
        return ("partition %s using %s%s(%s) as %s in %s" %
                (self.region,
                 self.cf_name,
                 ("<" + ", ".join(self.cf_params) + ">" if len(self.cf_params) > 0 else ""),
                 self.cf_args,
                 ", ".join(self.subregions),
                 self.body))

class PackExpr(Expr):
    def __init__(self, body, rrtype, rrparams):
        self.body = body
        self.rrtype = rrtype
        self.rrparams = rrparams

    def __str__(self):
        return ("pack %s as %s(%s)" %
                (self.body,
                 self.rrtype,
                 ", ".join(self.rrparams)))

class UnpackExpr(Expr):
    def __init__(self, argexpr, name, rrtype, rrparams, body):
        self.argexpr = argexpr
        self.name = name
        self.rrtype = rrtype
        self.rrparams = rrparams
        self.body = body

    def __str__(self):
        return ("unpack %s\n    as %s : %s(%s)\n    in %s" %
                (self.argexpr,
                 self.name, self.rrtype, self.rrparams,
                 self.body))

class UpRegionExpr(Expr):
    def __init__(self, ptrexpr, region):
        self.ptrexpr = ptrexpr
        self.region = region

    def __str__(self):
        return ("upregion(%s, %s)" % (self.ptrexpr, self.region))

class DownRegionExpr(Expr):
    def __init__(self, ptrexpr, region):
        self.ptrexpr = ptrexpr
        self.region = region

    def __str__(self):
        return ("downregion(%s, %s)" % (self.ptrexpr, self.region))

class IsNullExpr(Expr):
    def __init__(self, argexpr):
        self.argexpr = argexpr

    def __str__(self):
        return ("isnull(%s)" % (self.argexpr,))

class NullConstExpr(Expr):
    def __init__(self):
        pass

    def __str__(self):
        return "null"

class IntConstExpr(Expr):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return ("%d" % (self.value,))

class BoolConstExpr(Expr):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        if self.value:
            return "true"
        else:
            return "false"
