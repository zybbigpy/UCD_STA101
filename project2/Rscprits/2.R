##### set work directory and load dataset ##### 
setwd("/home/xmy/STA 101/Projects/P2")
prostate <- read.csv("prostate.csv", header = TRUE)
head(prostate, n = 3)

##### load packages #####
library(ggplot2)
library(pROC)
library(EnvStats)
library(bestglm)
library(nnet)
library(LogisticDx)
library(asbio)

##### set default parameters #####
ppi = 600

##### rename columns of datasets #####
names(prostate) = c("Y", "X1", "X2", "X3", "X4", "X5", "X6", "X7")
head(prostate, n = 3)
summary(prostate)

##### define functions #####

##### prostate summary #####

##### preparation of data #####

##### model selection #####
empty.model = glm(Y~1, data=prostate, family = binomial(link=logit))
full.model = glm(Y~., data=prostate, family = binomial(link=logit))
### Forward stepwise
F.model = step(empty.model, scope = list(lower=empty.model,upper=full.model),trace = FALSE, direction = "forward", criteria = "AIC")
### Backward stepwise
B.model = step(full.model, scope = list(lower=empty.model,upper=full.model),trace = FALSE, direction = "backward", criteria = "AIC")
### Forward/Backward stepwise
FB.model = step(empty.model, scope = list(lower=empty.model,upper=full.model),trace = FALSE, direction = "both", criteria = "AIC")
### Backward/Forward stepwise
BF.model = step(full.model, scope = list(lower=empty.model,upper=full.model),trace = FALSE, direction = "both", criteria = "AIC")
### display selected models
F.model
B.model
FB.model
BF.model

##### final model #####
final.model = glm(Y~X1+X2+X4, data=prostate, family = binomial(link=logit))
summary(final.model)
the.betas = final.model$coefficients
the.betas
exp.betas = exp(the.betas)
names(exp.betas) = c("(Intercept)", "exp.X1", "exp.X2", "exp.X4")
exp.betas
# display final model
### CI for betas
alpha = 0.05
the.CI = confint(final.model, level = 1 - alpha)
the.CI
exp.CI = exp(the.CI)
rownames(exp.CI) = c("(Intercept)", "exp.X1", "exp.X2", "exp.X4")
exp.CI


##### if X2 can be dropped #####
model.A = final.model
model.0 = glm(Y~X1+X4, data=prostate, family = binomial(link=logit))
LLA = logLik(model.A)
LL0 = logLik(model.0)
pA = length(model.A$coefficients)
p0 = length(model.0$coefficients)
LR = -2*(LL0-LLA)
p.value = pchisq(LR, df = pA-p0, lower.tail = FALSE)
p.value
##### if X4 can be dropped #####
model.A = final.model
model.0 = glm(Y~X1+X2, data=prostate, family = binomial(link=logit))
LLA = logLik(model.A)
LL0 = logLik(model.0)
pA = length(model.A$coefficients)
p0 = length(model.0$coefficients)
LR = -2*(LL0-LLA)
p.value = pchisq(LR, df = pA-p0, lower.tail = FALSE)
p.value
# interpret p-value


##### Diagnostics #####
### Pearson residuals
good.stuff = dx(final.model)
pear.r = good.stuff$Pr
std.r = good.stuff$sPr
plot.name = "pearson_std_e"
png(plot.name, width=6*ppi, height=4*ppi, res=ppi)
hist(std.r, main = "Pearson Standardized Residuals")
dev.off()
cutoff.std = 3.0
good.stuff[abs(std.r)>cutoff.std]
### dfbeta
df.beta = good.stuff$dBhat
plot.name = "dfbeta"
png(plot.name, width=6*ppi, height=4*ppi, res=ppi)
plot(df.beta, main = "Index plot of the change in the Betas")
dev.off()
cutoff.beta = 0.50
good.stuff[df.beta>cutoff.beta]
### dchisq
change.pearson = good.stuff$dChisq
plot.name = "dchisq"
png(plot.name, width=6*ppi, height=4*ppi, res=ppi)
plot(change.pearson, main = "Index plot of the change in the Pearson's test-statistic")
dev.off()
cutoff.pearson = 15
good.stuff[change.pearson>cutoff.pearson]

##### remove outliers #####
new.prostate  = prostate
new.prostate$std.r = std.r
new.prostate$df.beta = df.beta
new.prostate$change.pearson = change.pearson
new.prostate = new.prostate[-which(df.beta>cutoff.beta),]
new.prostate = new.prostate[-which(change.pearson>cutoff.pearson),]
the.ratio = (length(prostate$Y)-length(new.prostate$Y))/length(prostate$Y)
the.ratio

##### final best model #####
final.model = glm(Y~X1+X2+X4, data=new.prostate, family = binomial(link=logit))
summary(final.model)
the.betas = final.model$coefficients
the.betas
exp.betas = exp(the.betas)
names(exp.betas) = c("(Intercept)", "exp.X1", "exp.X2", "exp.X4")
exp.betas
# display final model
### CI for betas
alpha = 0.05
the.CI = confint(final.model, level = 1 - alpha)
the.CI
exp.CI = exp(the.CI)
rownames(exp.CI) = c("(Intercept)", "exp.X1", "exp.X2", "exp.X4")
exp.CI

##### error matrix #####
pi.0=0.50
truth = new.prostate$Y
predicted = ifelse(fitted(final.model)>pi.0,1,0)
my.table = table(truth, predicted)
my.table
sens = sum(predicted == 1 & truth ==1)/sum(truth ==1)
spec = sum(predicted == 0 & truth ==0)/sum(truth ==0)
error = sum(predicted!=truth)/length(predicted)
results = c(sens,spec,error)
names(results) = c("Sensitivity","Specificity","Error-Rate")
results
# interpret error matrix

##### ROC and AUC #####
plot.name = "auc.png"
png(plot.name, width=6*ppi, height=4*ppi, res=ppi)
my.auc = auc(final.model$y, fitted(final.model), plot = TRUE)
dev.off()
my.auc
auc.CI = ci(my.auc, level = 0.95)
auc.CI
# interpret auc.CI

##### predictive power #####
r = cor(final.model$y, final.model$fitted.values)
r
prop.red = 1-sum((final.model$y - final.model$fitted.values)^2)/sum((final.model$y - mean(final.model$y))^2)
prop.red
# interpret predictive power

##### predict #####
x.star = data.frame(X1 = 10, X2 = 5, X4 = 67)
the.predict = predict(final.model, x.star, type = "response")
the.predict

