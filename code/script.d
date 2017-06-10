import std.algorithm, std.stdio, std.string, std.array, std.conv;
import std.math : exp, pow, log, sqrt, abs;
import std.range;
import std.typecons : Tuple, tuple;

// dmd script.d -L-lopenblas -L-lpthread -L-llapacke -L-llapack -L-lm && ./script

auto read_yX(char sep = ',')(string name)
{
	auto file = File(name, "r");
	// Reads data into a double[][] array
	auto data = file.byLine(KeepTerminator.no)
	                .map!(a => a.split(sep)
	                	.map!(b => to!double(b))
	                	.array())
	                .array();
	auto y = data.map!(a => a[0]).array();
	auto x = data.map!(a => a[1 .. $]).array();
	return tuple(y, x);
}

enum CBLAS_LAYOUT {
	CblasRowMajor = 101,
	CblasColMajor = 102
}

alias CBLAS_LAYOUT.CblasRowMajor CblasRowMajor;
alias CBLAS_LAYOUT.CblasColMajor CblasColMajor;

enum CBLAS_TRANSPOSE {
	CblasNoTrans = 111,
	CblasTrans = 112,
    CblasConjTrans = 113
}

alias CBLAS_TRANSPOSE.CblasNoTrans CblasNoTrans;
alias CBLAS_TRANSPOSE.CblasTrans CblasTrans;
alias CBLAS_TRANSPOSE.CblasConjTrans CblasConjTrans;

// CBLAS & Lapacke functions
extern(C) @nogc nothrow{
void cblas_dgemm(in CBLAS_LAYOUT layout, in CBLAS_TRANSPOSE TransA,
                 in CBLAS_TRANSPOSE TransB, in int M, in int N,
                 in int K, in double alpha, in double  *A,
                 in int lda, in double  *B, in int ldb,
                 in double beta, double  *C, in int ldc);

alias int lapack_int;
alias int MKL_INT;
/* See IBM ESSL documentation for more details */
lapack_int LAPACKE_dgetrf(int matrix_layout, lapack_int m,
	        lapack_int n, double* a, lapack_int lda, 
	        lapack_int* ipiv);
lapack_int LAPACKE_dgetri(int matrix_layout, lapack_int n, 
	        double* a, lapack_int lda, in lapack_int* ipiv);
/* Function for calculating the norm of a vector */
double cblas_dnrm2 (in MKL_INT n , in double* x , in MKL_INT incx);
}

enum LAPACK_ROW_MAJOR = 101;
enum LAPACK_COL_MAJOR = 102;

/* Norm2 function */
double norm2(int incr = 1)(double[] x)
{
	return cblas_dnrm2(cast(int)x.length, x.ptr , incr);
}

/* Function for in place inverse of a matrix */
void inverse(int layout = LAPACK_ROW_MAJOR)(ref double[] matrix){
	int p = cast(int)sqrt(cast(double)matrix.length);
	int[] ipiv; ipiv.length = p;
	int info = LAPACKE_dgetrf(layout, p,
	            p, matrix.ptr, p, ipiv.ptr);
	assert(info == 0, "Illegal value from function LAPACKE_dgetrf");
	info = LAPACKE_dgetri(layout, p, 
	            matrix.ptr, p, ipiv.ptr);
	assert(info == 0, "Illegal value from function LAPACKE_dgetri");
}

/* Matrix multiplication */
double[] matmult(CBLAS_LAYOUT layout = CblasRowMajor, CBLAS_TRANSPOSE transA = CblasNoTrans, 
	         CBLAS_TRANSPOSE transB = CblasNoTrans)
             (double[] a, double[] b, int[2] dimA, int[2] dimB){
	         
	         double alpha = 1.;
	         double beta = 0.;
	         int m = transA == CblasNoTrans ? dimA[0] : dimA[1];
	         int n = transB == CblasNoTrans ? dimB[1] : dimB[0];
	         double[] c; //set this length
	         
	         int k = transA == CblasNoTrans ? dimA[1] : dimA[0];
	         
	         int lda, ldb, ldc;
	         if(transA == CblasNoTrans)
	         	lda = layout == CblasRowMajor? k: m;
	         else
	         	lda = layout == CblasRowMajor? m: k;
	         
	         if(transB == CblasNoTrans)
	         	ldb = layout == CblasRowMajor? n: k;
	         else
	         	ldb = layout == CblasRowMajor? k: n;

	         ldc = layout == CblasRowMajor ? n : m;
	         c.length = layout == CblasRowMajor ? ldc*m : ldc*n;

	         cblas_dgemm(layout, transA, transB, m, n,
                 k, alpha, a.ptr, lda, b.ptr, ldb,
                 beta, c.ptr, ldc);
	         return c;
}

// Repeated code ...
mixin template gParamCalcs()
{
	auto n = x.length, p = x[0].length;
	auto xB = zip(pars.repeat().take(n).array(), x)
	           .map!(a => zip(a[0], a[1])
	           	            .map!(a => a[0]*a[1])
	           	            .reduce!((a, b) => a + b))
	           .array();
	auto mu = xB.map!(a => exp(a)).array();
	auto k = (n - p)/zip(y, mu)
	                .map!(a => pow(a[0] - a[1], 2)/pow(a[1], 2))
	                .reduce!((a, b) => a + b);
}

/* This is a loglik for gamma distribution (omits gamma term) ... */
T gLogLik(T)(T[] pars, T[][] x, T[] y)
{
	mixin gParamCalcs;
	auto ll = zip(k.repeat().take(n).array(), xB, y, mu)
	          .map!(a => -a[0]*a[1] + a[0]*log(a[0]) + 
	          	          (a[0] - 1)*log(a[2]) - a[2]*a[0]/a[3])
	          .reduce!((a, b) => a + b);
	return ll;
}

T[] gGradient(T)(T[] pars, T[][] x, T[] y)
{
	mixin gParamCalcs;
	auto output = zip(k.repeat().take(n).array(), x, y, mu)
	            .map!(a => 
	            	zip(a[0].repeat().take(a[1].length),
	            		a[1],
	            		a[2].repeat().take(a[1].length),
	            		a[3].repeat().take(a[1].length))
	            	.map!(a => -a[0]*a[1] + a[1]*a[0]*(a[2]/a[3]))
	            	.array())
	            .reduce!((a, b) => zip(a, b).map!(a => a[0] + a[1]).array())
	            .array();
	return output;
}

/* Hessian */
auto gCurvature(T)(T[] pars, T[][] x, T[] y)
{
	mixin gParamCalcs;
	auto xPrime = zip(k.repeat().take(n).array(), x, y, mu)
	                .map!( a => 
	                	zip(a[0].repeat().take(a[1].length),
	            		    a[1],
	            		    a[2].repeat().take(a[1].length),
	            		    a[3].repeat().take(a[1].length))
	            		.map!(a => -a[2]*(a[0]/a[3])*a[1])
	            		.array())
	                .array();
    int[2] dims = [cast(int)n, cast(int)p];
	T[] curv = matmult!(CblasRowMajor, CblasTrans, CblasNoTrans)
             (xPrime.reduce!((a, b) => a ~ b).array(),
             	x.reduce!((a, b) => a ~ b).array(), 
             	dims, dims);
	return(curv);
}

T[] newtonRaphson(T, alias objFun = gLogLik,
	                 alias gradFun = gGradient,
	                 alias curveFun = gCurvature)
                     (T[] pars, T[][] x, T[] y,
                     	T tol = 1e-5, T ltol = 1e-5)
{
	T[] parsPrev = pars.dup;
	T ll = objFun(pars, x, y), llPrev = ll, llDiff = 1., parsDiff = 1.;
	int info, ipiv, n = cast(int)x.length, p = cast(int)x[0].length;
	T[] grad, curv, delta = T(0).repeat().take(p).array();
	while((parsDiff > tol) | (llDiff > ltol))
	{
		// Calculate and invert the hessian
		curv = curveFun(parsPrev, x, y);
		inverse(curv);
		// Calculate the gradient
		grad = gradFun(parsPrev, x, y);
		// Multiply the gradient by the hessian
		delta = matmult!(CblasRowMajor, CblasNoTrans, CblasNoTrans)
                        (curv, grad, [p, p], [p, 1]);
		// Updating the parameter
		pars = zip(pars, delta)
		          .map!(a => a[0] - a[1])
		          .array();
		parsDiff = zip(pars, parsPrev)
		                .map!(a => pow((a[0] - a[1]), 2.))
		                .reduce!((a, b) => a + b)
		                .sqrt;
		ll = objFun(pars, x, y);
		llDiff = abs(ll - llPrev);
		parsPrev = pars;
		llPrev = ll;
	}
	return pars;
}

/* For normal distribution gradient descent */
T[] nGradient(T)(T[] pars, T[] x, T y)
{
	T xSum = y - zip(pars, x).map!(a => a[0]*a[1])
	                   .reduce!((a, b) => a + b);
	T[] output = x.map!(a => a*xSum).array();
	return output;
}

T[] gradientDescent(T, alias gradFun = nGradient)
                     (T[] pars, T[][] x, T[] y, T alpha = 0.01,
                     	int nepochs = 500, T ptol = 1e-5)
{
	auto prevPars = pars.dup;
	auto temp = pars.dup;
	for(int i = 0; i < nepochs; ++i)
	{
		for(int j = 0; j < x.length; ++j)
		{
			prevPars = pars.dup;
			temp = gradFun(pars, x[j], y[j]);
			pars = zip(pars, temp)
			          .map!(a => a[0] + alpha*a[1])
			          .array();
			temp = zip(pars, prevPars)
			          .map!(a => a[0] - a[1])
			          .array();
			auto norm = norm2(temp);
			assert(norm != T.infinity, "infinite parameter issue");
			if(norm < ptol)
				break;
		}
	}
	return pars;
}


void main()
{
	auto z = read_yX("freeny.csv");
	double[][] x;
	double[] y;
	y = z[0]; x = z[1];
	writeln("Output: ", newtonRaphson([1.,1.,1.,1.,1.], x, y));
    z = read_yX("ex3.csv");
	y = z[0]; x = z[1];
	writeln("Output: ", gradientDescent([0.,0.,0.], x, y, 0.01, 500));
}


