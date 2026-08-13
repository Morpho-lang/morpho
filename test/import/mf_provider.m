// Provider for metafunction import tests

fn f(String x) {
    return "string"
}

fn f(Int x) {
    return "int"
}

fn helper(x) {
    return f(x)
}

fn _privateFn() {
    return "secret"
}

class Foo {
    _x() {
        return "method"
    }
}
