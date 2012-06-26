# AST nodes for python parser

class DuplicateTypeDefinition(Exception):
    pass

class DuplicateTaskDefinition(Exception):
    pass

class TypeError(Exception):
    pass

class UnimplementedTypeCheck(TypeError):
    def __init__(self, obj):
        self.obj = obj

    def __str__(self):
        return "type checking not implemented for class " + self.obj.__class__.__name__

class IsNullArgTypeError(TypeError):
    def __init__(self, expr, argtype):
        self.expr = expr
        self.argtype = argtype

    def __str__(self):
        return "isnull requires a pointer argument - got: %s" % (self.argtype,)

class NoReadPriviledgeError(TypeError):
    def __init__(self, expr, ptrtype, privs):
        self.expr = expr
        self.ptrtype = ptrtype
        self.privs = privs

    def __str__(self):
        return "read of %s attempted with the following priviledges:\n%s" % (self.ptrtype, self.privs)

class NoWritePriviledgeError(TypeError):
    def __init__(self, expr, ptrtype, privs):
        self.expr = expr
        self.ptrtype = ptrtype
        self.privs = privs

    def __str__(self):
        return "write of %s attempted with the following priviledges:\n%s" % (self.ptrtype, self.privs)

class NoReducePriviledgeError(TypeError):
    def __init__(self, expr, ptrtype, privs):
        self.expr = expr
        self.ptrtype = ptrtype
        self.privs = privs

    def __str__(self):
        return "reduce to %s attempted with the following priviledges:\n%s" % (self.ptrtype, self.privs)

class LetInitTypeError(TypeError):
    def __init__(self, expr, exptype, acttype):
        self.expr = expr
        self.exptype = exptype
        self.acttype = acttype

    def __str__(self):
        return ("expected type %s for let initialization, got %s" %
                (self.exptype,
                 self.acttype))

class WrongTypeClassError(TypeError):
    def __init__(self, expr, expclass, acttype):
        self.expr = expr
        self.expclass = expclass
        self.acttype = acttype

    def __str__(self):
        return ("expected a %s type, got: %s" % (self.expclass, self.acttype))

class PackTypeMismatchError(TypeError):
    def __init__(self, expr, exptype, acttype):
        self.expr = expr
        self.exptype = exptype
        self.acttype = acttype

    def __str__(self):
        return ("type mismatch in pack - wanted %s, got %s" %
                (self.exptype, self.acttype))

class UnpackTypeMismatchError(TypeError):
    def __init__(self, expr, exptype, acttype):
        self.expr = expr
        self.exptype = exptype
        self.acttype = acttype

    def __str__(self):
        return ("type mismatch in unpack - wanted %s, got %s" %
                (self.exptype, self.acttype))

class WriteTypeMismatchError(TypeError):
    def __init__(self, expr, ptrtype, valtype):
        self.expr = expr
        self.ptrtype = ptrtype
        self.valtype = valtype

    def __str__(self):
        return ("can't write a value of type %s to a pointer of type %s" %
                (self.valtype, self.ptrtype))

class ReduceTypeMismatchError(TypeError):
    def __init__(self, expr, ptrtype, valtype):
        self.expr = expr
        self.ptrtype = ptrtype
        self.valtype = valtype

    def __str__(self):
        return ("can't reduce a value of type %s to a pointer of type %s" %
                (self.valtype, self.ptrtype))

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

    def type_check(self):
        for t in self.tasks.itervalues():
            try:
                t.type_check(self)
            except TypeError as e:
                print "type checking of %s failed: %s" % (t.name, e)

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

    def type_check(self, pgrm):
        env = dict(self.formals.byname)
        privs = self.effects
        consts = set()  # no constraints initially
        bodytype = self.body.get_type(pgrm, env, privs, consts)
        print "type of task %s body = %s" % (self.name, bodytype)

class Type:
    pass

class IntType(Type):
    def __init__(self):
        pass

    def __str__(self):
        return "int"

    def rename(self, bindings):
        return self

    def equals(self, other, **kwargs):
        return isinstance(other, IntType)

class BoolType(Type):
    def __init__(self):
        pass

    def __str__(self):
        return "bool"

    def rename(self, bindings):
        return self

    def equals(self, other, **kwargs):
        return isinstance(other, BoolType)

class PtrType(Type):
    def __init__(self, elemtype, regions):
        self.elemtype = elemtype
        self.regions = regions

    def __str__(self):
        if len(self.regions) == 1:
            return ("%s@%s" % (self.elemtype, self.regions[0]))
        else:
            return ("%s@(%s)" % (self.elemtype,
                                 ",".join(self.regions)))

    def rename(self, bindings):
        return PtrType(self.elemtype.rename(bindings),
                       [ bindings.get(x, x) for x in self.regions ])
                       #bindings.get(self.region))

    def isnullptrtype(self):
        return (self.elemtype is None)

    def equals(self, other, **kwargs):
        if not isinstance(other, PtrType):
            return False
        if self.isnullptrtype():
            if other.isnullptrtype():
                return True
            else:
                return kwargs.get("nullok", True)
        else:
            if other.isnullptrtype():
                return kwargs.get("nullok", True)
            else:
                if not self.elemtype.equals(other.elemtype, **kwargs):
                    return False
                if len(self.regions) <> len(other.regions):
                    return False
                for r1,r2 in zip(self.regions, other.regions):
                    if r1 <> r2:
                        return False
                return True

class TupleType(Type):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def rename(self, bindings):
        return TupleType(self.lhs.rename(bindings), 
                         self.rhs.rename(bindings))

    def equals(self, other, **kwargs):
        if not isinstance(other, TupleType):
            return False
        return (self.lhs.equals(other.lhs, **kwargs) and
                self.rhs.equals(other.rhs, **kwargs))

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

    def rename(self, bindings):
        return UserType(self.name,
                        [ bindings.get(x, x) for x in self.params ])
                        #[ bindings.get(x) for x in self.params ])

    # returns the expanded version of the type
    def expand(self, prgm):
        if self.name not in prgm.types:
            raise UnknownTypeError(self)
        utype = prgm.types[self.name]
        if len(self.params) <> len(utype.params):
            raise TypeParamCountError(self, utype)
        return utype.innertype.rename(dict(zip(utype.params, self.params)))

    def equals(self, other, **kwargs):
        # quick way - if both are the same usertype, just compare params
        if isinstance(other, UserType) and (self.name == other.name):
            for sp, op in zip(self.params, other.params):
                if sp <> op:
                    return False
            return True

        # slower way - expand one/both of our types and recompare
        selfexp = self.expand(kwargs.get("pgrm"))
        print ("self: expanded %s to %s" % (self, selfexp))
        if isinstance(other, UserType):
            otherexp = other.expand(kwargs.get("pgrm"))
            print ("other: expanded %s to %s" % (other, otherexp))
        else:
            otherexp = other
        return selfexp.equals(otherexp, **kwargs)

class RRType(Type):
    def __init__(self, regions, innertype, constraints):
        self.regions = regions
        self.innertype = innertype
        self.constraints = constraints

    def __str__(self):
        return ("rr[%s] %s where %s" % (", ".join(self.regions),
                                        self.innertype,
                                        " and ".join(map(str, self.constraints))))

    def rename(self, bindings):
        newbindings = dict(bindings)
        for r in self.regions:
            if r in newbindings:
                raise RegionConflictError()
            #newbindings[r] = r
        return RRType(self.regions,
                      self.innertype.rename(newbindings),
                      [ c.rename(newbindings) for c in self.constraints ])

    def equals(self, other, **kwargs):
        if not isinstance(other, RRType):
            return False

        # step 1: number of regions bound
        if len(self.regions) <> len(other.regions):
            return False

        # now see if the inner types are the same up to renaming
        return self.innertype.equals(other.innertype.rename(dict(zip(other.regions,
                                                                     self.regions))),
                                     **kwargs)

class RegionConstraints:
    def __init__(self):
        self.regions = dict()

    def populate(self, *list):
        for r in list:
            if r in self.regions: continue
            self.regions[r] = dict(subs = set(),
                                   supers = set(),
                                   disjoints = set())

    def add_subregion(self, rc, rp):
        self.populate(rc, rp)
        # connect rc and all its children to rp and all its parents
        subs = set(self.regions[rc]['subs'])
        subs.add(rc)
        supers = set(self.regions[rp]['supers'])
        supers.add(rp)
        for rc2 in subs:
            self.regions[rc2]['supers'] |= supers
        for rp2 in supers:
            self.regions[rp2]['subs'] |= subs
        return self

    def add_disjoint(self, rc, rp):
        self.populate(rc, rp)
        self.regions[rc]['disjoints'].add(rp)
        self.regions[rp]['disjoints'].add(rc)
        return self

class RegionConstraint:
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return ("%s %s %s" % (self.lhs, self.op, self.rhs))

    def rename(self, bindings):
        return RegionConstraint(self.op,
                                bindings.get(self.lhs, self.lhs),
                                bindings.get(self.rhs, self.rhs))

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

    # all expressions have the ability to return their type, given an
    # environment that maps names to types, a set of priviledges, and a 
    # set of constraints
    def get_type(self, pgrm, env, privs, consts):
        raise UnimplementedTypeCheck(self)

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

    def get_type(self, pgrm, env, privs, consts):
        # step 1: the value's expression must have the right type
        actvaltype = self.valexpr.get_type(pgrm, env, privs, consts)
        if not self.valtype.equals(actvaltype, pgrm = pgrm):
            raise LetInitTypeError(self, self.valtype, actvaltype)

        # step 2: type check the body with an updated environment
        newenv = dict(env)
        newenv[self.valname] = self.valtype

        return self.body.get_type(pgrm, newenv, privs, consts)

class IdentExpr(Expr):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def get_type(self, pgrm, env, privs, consts):
        # an ident has to be in the environment
        if self.name in env:
            return env[self.name]
        else:
            raise UnknownIdentifierTypeError(self, env, privs, consts)

class TupleExpr(Expr):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return ("<%s, %s>" % (self.lhs, self.rhs))

    def get_type(self, pgrm, env, privs, consts):
        lhstype = self.lhs.get_type(pgrm, env, privs, consts)
        rhstype = self.rhs.get_type(pgrm, env, privs, consts)
        return TupleType(lhs = lhstype, rhs = rhstype)

class FieldExpr(Expr):
    def __init__(self, subexpr, field):
        self.subexpr = subexpr
        self.field = field

    def __str__(self):
        return ("%s.%d" % (self.subexpr, self.field))

    def get_type(self, pgrm, env, privs, consts):
        # step 1: get subexpression's type, make sure it's a tuple
        #  (might have to expand a user type to get there
        exprtype = self.subexpr.get_type(pgrm, env, privs, consts)
        if isinstance(exprtype, UserType):
            exprtype = exprtype.expand(pgrm)
        if not isinstance(exprtype, TupleType):
            raise WrongTypeClassError(self, "tuple", exprtype)

        # step 2: select the right piece based on the field number
        if self.field == 1:
            return exprtype.lhs
        if self.field == 2:
            return exprtype.rhs
        raise BadFieldNumberTypeError(self, self.field)

class ReadExpr(Expr):
    def __init__(self, ptrexpr):
        self.ptrexpr = ptrexpr

    def __str__(self):
        return ("read(%s)" % (self.ptrexpr,))

    def get_type(self, pgrm, env, privs, consts):
        # step 1: get the pointer's type
        ptrtype = self.ptrexpr.get_type(pgrm, env, privs, consts)
        if not isinstance(ptrtype, PtrType):
            raise NonPointerTypeError(self, ptrtype)

        # step 2: check priviledges
        for r in ptrtype.regions:
            if r not in privs.reads:
                raise NoReadPriviledgeError(self, ptrtype, privs)

        # step 3: result type is element type of pointer
        return ptrtype.elemtype

class WriteExpr(Expr):
    def __init__(self, ptrexpr, valexpr):
        self.ptrexpr = ptrexpr
        self.valexpr = valexpr

    def __str__(self):
        return ("write(%s, %s)" % (self.ptrexpr, self.valexpr))

    def get_type(self, pgrm, env, privs, consts):
        # step 1: get the pointer's type
        ptrtype = self.ptrexpr.get_type(pgrm, env, privs, consts)
        if not isinstance(ptrtype, PtrType):
            raise NonPointerTypeError(self, ptrtype)

        # step 2: check priviledges
        for r in ptrtype.regions:
            if r not in privs.writes:
                raise NoWritePriviledgeError(self, ptrtype, privs)

        # step 3: value type is element type of pointer
        valtype = self.valexpr.get_type(pgrm, env, privs, consts)
        if not ptrtype.elemtype.equals(valtype, pgrm = pgrm):
            raise WriteTypeMismatchError(self, ptrtype, valtype)

        # result of write is bool (an arbitrary choice)
        return BoolType()

class ReduceExpr(Expr):
    def __init__(self, func, ptrexpr, valexpr):
        self.func = func
        self.ptrexpr = ptrexpr
        self.valexpr = valexpr

    def __str__(self):
        return ("reduce(%s, %s, %s)" % (self.func, self.ptrexpr, self.valexpr))

    def get_type(self, pgrm, env, privs, consts):
        # step 1: get the pointer's type
        ptrtype = self.ptrexpr.get_type(pgrm, env, privs, consts)
        if not isinstance(ptrtype, PtrType):
            raise NonPointerTypeError(self, ptrtype)

        # step 2: check priviledges
        if self.func not in privs.reduces:
            raise NoReducePriviledgeError(self, ptrtype, privs)
        for r in ptrtype.regions:
            if r not in privs.reduces[self.func]:
                raise NoReducePriviledgeError(self, ptrtype, privs)

        # TODO: check function signature - needs to be monomorphic, with
        #  first arg and result that match pointer's element type, and
        #  second arg that matches valtype
        valtype = self.valexpr.get_type(pgrm, env, privs, consts)

        if not ptrtype.elemtype.equals(valtype, pgrm = pgrm):
            raise ReduceTypeMismatchError(self, ptrtype, valtype)

        # result of reduce is bool (an arbitrary choice)
        return BoolType()


class BinOpExpr(Expr):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return ("%s %s %s" % (self.lhs, self.op, self.rhs))

    def get_type(self, pgrm, env, privs, consts):
        lhstype = self.lhs.get_type(pgrm, env, privs, consts)
        rhstype = self.rhs.get_type(pgrm, env, privs, consts)

        # integer -> integer ops
        if (self.op == "+"):
            if not isinstance(lhstype, IntType):
                raise WrongTypeClassError(self, "int", lhstype)
            if not isinstance(rhstype, IntType):
                raise WrongTypeClassError(self, "int", rhstype)
            return IntType()

        # integer -> boolean ops
        if (self.op == "<") or (self.op == ">"):
            if not isinstance(lhstype, IntType):
                raise WrongTypeClassError(self, "int", lhstype)
            if not isinstance(rhstype, IntType):
                raise WrongTypeClassError(self, "int", rhstype)
            return BoolType()

        # boolean -> boolean ops
        if (self.op == "|") or (self.op == "&"):
            if not isinstance(lhstype, BoolType):
                raise WrongTypeClassError(self, "bool", lhstype)
            if not isinstance(rhstype, BoolType):
                raise WrongTypeClassError(self, "bool", rhstype)
            return BoolType()

        print self.op
        raise BadBadBad(op)

class IfExpr(Expr):
    def __init__(self, condexpr, thenexpr, elseexpr):
        self.condexpr = condexpr
        self.thenexpr = thenexpr
        self.elseexpr = elseexpr

    def __str__(self):
        return ("if %s then %s else %s" %
                (self.condexpr, self.thenexpr, self.elseexpr))

    def get_type(self, pgrm, env, privs, consts):
        # step 1: condition expression must be a bool
        condtype = self.condexpr.get_type(pgrm, env, privs, consts)
        if not isinstance(condtype, BoolType):
            raise NonBoolConditionError(self, condtype)

        # step 2: then and else must have same type, which is our type
        thentype = self.thenexpr.get_type(pgrm, env, privs, consts)
        elsetype = self.elseexpr.get_type(pgrm, env, privs, consts)

        if not thentype.equals(elsetype, pgrm = pgrm):
            raise IfTypeMismatchError(self, thentype, elsetype)

        # special case: if then side is a null pointer, use else
        if isinstance(thentype, PtrType) and thentype.isnullptrtype():
            return elsetype
        else:
            return thentype

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

    def get_type(self, pgrm, env, privs, consts):
        # step 1: find the task we're calling
        if self.name not in pgrm.tasks:
            raise UnknownTaskError(self, pgrm.tasks)
        task = pgrm.tasks[self.name]

        # step 2: check param count, make binding dictionary
        if len(self.params) <> len(task.params):
            raise TaskParamCountError(self, task)
        bindings = dict(zip(task.params, self.params))

        # step 3: check argument count and types
        if len(self.args) <> len(task.formals.byorder):
            raise TaskArgCountError(self, task)
        for i, n in enumerate(task.formals.byorder):
            exptype = task.formals.byname[n].rename(bindings)
            acttype = self.args[i].get_type(pgrm, env, privs, consts)
            #print "%s: %s vs %s" % (n, exptype, acttype)
            if not exptype.equals(acttype, pgrm = pgrm):
                raise TaskArgTypeMismatchError(self, task, n, exptype, acttype)

        # step 4: check effects against constraints
        # TODO

        # result type is properly-renamed result type of task
        return task.rettype.rename(bindings)

class NewExpr(Expr):
    def __init__(self, ptrtype):
        self.ptrtype = ptrtype

    def __str__(self):
        return ("new %s" % (self.ptrtype,))

    def get_type(self, pgrm, env, privs, consts):
        # check to make sure we actually have a pointer type - if we do,
        #  that's our type
        if not isinstance(self.ptrtype, PtrType):
            raise WrongTypeClassError(self, "pointer", self.ptrtype)
        return self.ptrtype

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

    def get_type(self, pgrm, env, privs, consts):
        # TODO: check partitioning function args/privs

        # add constraints that each subregion is a subset of the original,
        #  and that they are disjoint from each other
        newconsts = list(consts)
        for i, rs1 in enumerate(self.subregions):
            newconsts.append(RegionConstraint(op = "<=", 
                                              lhs = rs1, 
                                              rhs = self.region))
            for rs2 in self.subregions[i+1:]:
                newconsts.append(RegionConstraint(op = "*",
                                                  lhs = rs1, rhs = rs2))

        # now type-check body with these additional constraints
        return self.body.get_type(pgrm, env, privs, newconsts)

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

    def get_type(self, pgrm, env, privs, consts):
        # step 1: get the type of the argument and make sure it's an RR
        #  (and the right one)
        acttype = self.body.get_type(pgrm, env, privs, consts)

        exptype = self.rrtype
        if isinstance(exptype, UserType):
            exptype = exptype.expand(pgrm)
        if not isinstance(exptype, RRType):
            raise WrongTypeClassError(self, "RR", exptype)
        if len(self.rrparams) <> len(exptype.regions):
            raise RRParamCountError(self, self.rrparams, exptype)
        exptype = exptype.innertype.rename(dict(zip(exptype.regions, self.rrparams)))
        #print ("pack check:\n  %s\nvs\n  %s" % (exptype, acttype))
        if not acttype.equals(exptype, pgrm = pgrm):
            raise PackTypeMismatchError(self, exptype, acttype)

        return self.rrtype

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

    def get_type(self, pgrm, env, privs, consts):
        # step 1: get the type of the argument and make sure it's an RR
        #  (and the right one)
        acttype = self.argexpr.get_type(pgrm, env, privs, consts)
        if isinstance(acttype, UserType):
            acttype = acttype.expand(pgrm)
        if not isinstance(acttype, RRType):
            raise WrongTypeClassError(self, "RR", acttype)

        exptype = self.rrtype
        if isinstance(exptype, UserType):
            exptype = exptype.expand(pgrm)
        if not isinstance(exptype, RRType):
            raise WrongTypeClassError(self, "RR", exptype)
        if not acttype.equals(exptype, pgrm = pgrm):
            raise UnpackTypeMismatchError(self, exptype, acttype)

        # now expand the unpacked type with the new names
        if len(self.rrparams) <> len(exptype.regions):
            raise RRParamCountError(self, self.rrparams, exptype)
        exptype = exptype.innertype.rename(dict(zip(exptype.regions, self.rrparams)))

        # update the environment to give the new name the new type
        # TODO: rename any existing (now shadowed) regions with the same name in any type
        newenv = dict(env)
        newenv[self.name] = exptype
        # TODO: also add the constraints we gained
        newconsts = consts
        
        # type of unpack is type of body
        return self.body.get_type(pgrm, newenv, privs, newconsts)

class UpRegionExpr(Expr):
    def __init__(self, ptrexpr, regions):
        self.ptrexpr = ptrexpr
        self.regions = regions

    def __str__(self):
        return ("upregion(%s, %s)" % (self.ptrexpr, 
                                      (", ".join(self.regions))))

    def get_type(self, pgrm, env, privs, consts):
        # step 1: pointer must be a pointer type
        ptrtype = self.ptrexpr.get_type(pgrm, env, privs, consts)
        if not isinstance(ptrtype, PtrType):
            raise WrongTypeClassError(self, "pointer", ptrtype)

        # step 2: must be able to show that the pointer's region is a
        #  a subregion of the one to which we're upcasting
        # TODO

        # result pointer type uses new region
        return PtrType(elemtype = ptrtype.elemtype, regions = self.regions)

class DownRegionExpr(Expr):
    def __init__(self, ptrexpr, regions):
        self.ptrexpr = ptrexpr
        self.regions = regions

    def __str__(self):
        return ("downregion(%s, %s)" % (self.ptrexpr, 
                                        (", ".join(self.regions))))

    def get_type(self, pgrm, env, privs, consts):
        # step 1: pointer must be a pointer type
        ptrtype = self.ptrexpr.get_type(pgrm, env, privs, consts)
        if not isinstance(ptrtype, PtrType):
            raise WrongTypeClassError(self, "pointer", ptrtype)

        # step 2: must be able to show that the target region is statically
        #  known to be a subregion of the pointer's region
        # TODO

        # result pointer type uses new region
        return PtrType(elemtype = ptrtype.elemtype, regions = self.regions)

class IsNullExpr(Expr):
    def __init__(self, argexpr):
        self.argexpr = argexpr

    def __str__(self):
        return ("isnull(%s)" % (self.argexpr,))

    def get_type(self, pgrm, env, privs, consts):
        # isnull requires that its argument be a pointer
        argtype = self.argexpr.get_type(pgrm, env, privs, consts)
        if not isinstance(argtype, PtrType):
            raise IsNullArgTypeError(self, argtype)

        return BoolType()

class NullConstExpr(Expr):
    def __init__(self):
        pass

    def __str__(self):
        return "null"

    def get_type(self, pgrm, env, privs, consts):
        # TODO: is there a better way to handle this?
        return PtrType(elemtype = None, regions = [ None ])

class IntConstExpr(Expr):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return ("%d" % (self.value,))

    def get_type(self, pgrm, env, privs, consts):
        return IntType()

class BoolConstExpr(Expr):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        if self.value:
            return "true"
        else:
            return "false"

    def get_type(self, pgrm, env, privs, consts):
        return BoolType()
