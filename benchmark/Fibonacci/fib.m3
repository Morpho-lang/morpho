// Tests functions and function calls
f(x) = {
	if(x<2, return(x));
    return(f(x-1)+f(x-2));
}

timing(
	a = f(28);
)

print(a)
