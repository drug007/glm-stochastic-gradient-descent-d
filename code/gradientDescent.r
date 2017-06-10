#=================================================================================================
# Linear Regression by Stochastic Gradient Descent
#=================================================================================================

# Y variable
y <- read.table("https://raw.githubusercontent.com/dbouquin/IS_605/master/sgd_ex_data/ex3y.dat")
y <- as.matrix(y)
colnames(y) <- NULL


# X variable
X <- read.table("https://raw.githubusercontent.com/dbouquin/IS_605/master/sgd_ex_data/ex3x.dat")
X <- as.matrix(X)
colnames(X) <- NULL

normalize <- function(x)
    (x - min(x))/(max(x) - min(x))

# randomize the data
set.seed(0)
indexes <- sample(nrow(X))

# Randomize indices
X <- X[indexes, , drop = FALSE]
y <- y[indexes, , drop = FALSE]

X <- apply(X, 2, normalize)
X <- cbind(1, X)

# Save X, y to file ...
save(X, y, file = file.path("ex3.RData"))

ex3 <- cbind(y, X)
write.table(ex3, file = file.path("ex3.csv"), row.names = FALSE, col.names = FALSE, sep = ",")

# Gradient, pars and y are column matrices
nGradient <- function(pars, x, y)
{
	t(x) %*% (y - (x %*% pars))
}

nGradientDescent <- function(pars, x, y, gradFun = nGradient, alpha = 0.01, ptol = 1E-5, nepochs = 500)
{
	prevPars <- pars

	for(i in 1:nepochs)
	{
		for(j in 1:nrow(x))
		{
			prevPars <- pars
			pars <- pars + alpha*gradFun(pars, x[j, , drop = FALSE], y[j, , drop = FALSE])

			if(is.infinite(norm(pars - prevPars)))
			    stop("infinite parameter problems")
			if(norm(pars - prevPars) < ptol)
			{
			    break
			}
		}
	}
	return(pars)
}


#=================================================================================================
nGradientDescent(matrix(rep(0, 3)), X, y, alpha = 0.01)
#=================================================================================================
#=================================================================================================
# Linear regression solve
solve(t(X)%*%X) %*% t(X)%*%y
#=================================================================================================
