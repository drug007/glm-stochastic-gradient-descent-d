#===================================
#  Newton Raphson attempt for GLM
#===================================

require(datasets)

X <- model.matrix(y ~ ., data = freeny)
y <- matrix(freeny$y, nc = 1)

normalize <- function(x)
    (x - min(x))/(max(x) - min(x))

# randomize the data
set.seed(0)
indexes <- sample(nrow(X))

# Randomize indices
X <- X[indexes, , drop = FALSE]
y <- y[indexes, , drop = FALSE]

X <- apply(X, 2, normalize)
X[,1] <- 1

yX <- cbind(y, X)
dimnames(yX) <- NULL

if(FALSE){
	write.table(yX, "/home/chib/code/D/blogCode/glm-stochastic-gradient-descent/script/LR/freeny.csv",
	      col.names = FALSE, row.names = FALSE, sep = ",")
}

# Modified Log likelihood because of problems with gamma function ...
gObjective <- function(pars, x, y)
{
	xB <- x %*% pars
	mu <- exp(xB)
	k <- 1/((sum(((y - mu)^2)/(mu^2))/(nrow(x) - ncol(x))))
	logLikelihood <- - k*xB + k*log(k) + (k - 1)*log(y) - y*k/mu # - log(gamma(k))
	logLikelihood <- sum(logLikelihood)
	return(logLikelihood)
}

# Gradient
gGradient <- function(pars, x, y)
{
	xB <- x %*% pars
	mu <- exp(xB)
	k <- 1/((sum(((y - mu)^2)/(mu^2))/(nrow(x) - ncol(x))))
	gradient <- -k*x + as.vector(k*(y/mu))*x
	gradient <- apply(gradient, 2, sum)
	return(gradient)
}

# Hessian
gCurvature <- function(pars, x, y)
{
	xB <- x %*% pars
	mu <- exp(xB)
	k <- 1/((sum(((y - mu)^2)/(mu^2))/(nrow(x) - ncol(x))))
	curvature <- -t(as.vector(y*k/mu)*x) %*% x
	return(curvature)
}

# Newton Raphson implementation for GLM solver
newtonRaphson <- function(x, y, pars, objFun = gObjective, gradFun = gGradient, curveFun = gCurvature, tol = 1E-5, ltol = 1E-5)
{
	if(missing(pars))
	{
		pars <- rep(0, ncol(x))
		parsPrev <- pars
	}else{
		parsPrev <- pars
	}
	parsDiff <- 1
	logLikDiff <- 1
	logLikelihood <- objFun(pars, x, y)
	logLikelihoodPrev <- logLikelihood
	while(parsDiff > tol && logLikDiff > ltol)
	{
		inverse = solve(curveFun(parsPrev, x, y))
		delta <- inverse %*% gradFun(parsPrev, x, y)
		pars <- parsPrev - delta
		parsDiff <- pars - parsPrev
		parsDiff <- sqrt(t(parsDiff) %*% parsDiff)
		logLikelihood <- objFun(pars, x, y)
		logLikDiff <- abs(logLikelihood - logLikelihoodPrev)
		parsPrev <- pars
		logLikelihoodPrev <- logLikelihood
	}
	return(pars)
}

# Gradient Descent implementation for GLM solver
gradientDescent <- function(x, y, pars, objFun = gObjective, gradFun = gGradient, lRate = 0.01, tol = 1E-5, ltol = 1E-5)
{
	if(missing(pars))
	{
		pars <- rep(0, ncol(x))
		parsPrev <- pars
	}else{
		parsPrev <- pars
	}
	parsDiff <- 1
	logLikDiff <- 1
	logLikelihood <- objFun(pars, x, y)
	logLikelihoodPrev <- logLikelihood
	while(parsDiff > tol && logLikDiff > ltol)
	{
		delta <- lRate*matrix(gradFun(parsPrev, x, y))/nrow(x)
		pars <- parsPrev - delta
		parsDiff <- pars - parsPrev
		parsDiff <- sqrt(t(parsDiff) %*% parsDiff)
		logLikelihood <- objFun(pars, x, y)
		logLikDiff <- abs(logLikelihood - logLikelihoodPrev)
		parsPrev <- pars
		logLikelihoodPrev <- logLikelihood
	}
	return(pars)
}

# Newton Raphson Works
nrModel <- newtonRaphson(X, y)
glmModel <- matrix(glm.fit(x = X, y = y, family = Gamma("log"))$coeff)
