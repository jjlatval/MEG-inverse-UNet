library(stargazer)
library(Hmisc)

cwd <- getwd()
path <- paste(cwd, "/network_tests/results_processed.csv", sep="")

data <- read.csv(path, header=TRUE, sep=",", dec=".", fileEncoding="latin1")

data[,6] <- as.double(data[,6])
data[,7] <- as.double(data[,7])
data[,8] <- as.double(data[,8])
data[,9] <- as.double(data[,9])
data[,10] <- as.double(data[,10])
data[,11] <- as.double(data[,11])
data[,12] <- as.double(data[,12])
data[,13] <- as.double(data[,13])
data[,14] <- as.double(data[,14])
data[,15] <- as.double(data[,15])
data[,16] <- as.double(data[,16])
data[,17] <- as.double(data[,17])

sink("summary.txt")
print(summary(data))
sink()

sink("summary_table.txt")
t <- stargazer(data[,6:13]*100)
t2 <- stargazer(data[,14:17])
print(t)
print(t2)
sink()

dle <- boxplot(data[, 6:9]*100, ylab="DLE (cm)", names=c("MNE", "sLORETA", "dSPM", "U-Net"))

sd <- boxplot(data[, 10:13]*100, ylab=expression(SD ~ sqrt(cm)), names=c("MNE", "sLORETA", "dSPM", "U-Net"))

oa <- boxplot(data[, 14:17], ylab="amplitude", names=c("MNE", "sLORETA", "dSPM", "U-Net"))

dens1 <- density(data[, 6]*100)
dens2 <- density(data[, 7]*100)
dens3 <- density(data[, 8]*100)
dens4 <- density(data[, 9]*100)
sd1 <- density(data[,10]*100)
sd2 <- density(data[,11]*100)
sd3 <- density(data[,12]*100)
sd4 <- density(data[,13]*100)
oa1 <- density(data[,14])
oa2 <- density(data[,15])
oa3 <- density(data[,16])
oa4 <- density(data[,17])

par(mfrow=c(2,2))

plot(dens1, main="MNE DLE")
plot(dens2, main="sLORETA DLE")
plot(dens3, main="dSPM DLE")
plot(dens4, main="U-Net DLE")

plot(sd1, main="MNE SD")
plot(sd2, main="sLORETA SD")
plot(sd3, main="dSPM SD")
plot(sd4, main="U-Net SD")

plot(oa1, main="MNE OA")
plot(oa2, main="sLORETA OA")
plot(oa3, main="dSPM OA")
plot(oa4, main="U-Net OA")

plot(oa1, main="MNE OA")
plot(oa2, main="sLORETA OA")
plot(oa3, main="dSPM OA")
plot(oa4, main="U-Net OA")

sink("covariances.txt")
cov1 <- cov(as.matrix(data[,6:9]))
cov2 <- cov(as.matrix(data[,10:13]))
cov3 <- cov(as.matrix(data[,14:17]))

print(cov1)
print(cov2)
print(cov3)
sink()

sink("correlations.txt")
corr1 <- rcorr(as.matrix(data[,6:9]))
corr2 <- rcorr(as.matrix(data[,10:13]))
corr3 <- rcorr(as.matrix(data[,14:17]))

# Spearman correlations
corr4 <- rcorr(as.matrix(data[,6:9]), type=c("spearman"))
corr5 <- rcorr(as.matrix(data[,10:13]), type=c("spearman"))
corr6 <- rcorr(as.matrix(data[,14:17]), type=c("spearman"))

print(corr1)
print(corr2)
print(corr3)
print(corr4)
print(corr5)
print(corr6)
sink()
