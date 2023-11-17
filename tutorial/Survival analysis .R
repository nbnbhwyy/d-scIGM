### Survival analysis 
library(cgdsr)
library(survival)
library(survminer)
mycgds <- CGDS('http://www.cbioportal.org/')
all_tcga_studies <- getCancerStudies(mycgds)
skcm_2015 <- 'skcm_tcga'
mycaselist <- getCaseLists(mycgds, skcm_2015)[6,1]
mygeneticprofile <- getGeneticProfiles(mycgds,skcm_2015)[7,1]

choose_genes <- c('CD3D', 'AKT1', 'LCK', 'TLR4', 'STAT1', 'STAT3', 'UBB', 'CD86')

expr <- getProfileData(mycgds, choose_genes, mygeneticprofile, mycaselist)
clinicaldata <- getClinicalData(mycgds, mycaselist)
dat <- clinicaldata[clinicaldata$OS_MONTHS > 0, ]
dat <- cbind(clinicaldata[, c('OS_STATUS', 'OS_MONTHS')], expr[rownames(clinicaldata), ])
dat <- dat[dat$OS_MONTHS > 0, ]
dat <- dat[!is.na(dat$OS_STATUS), ]
dat$OS_STATUS <- as.character(dat$OS_STATUS)
dat[, -(1:2)] <- apply(dat[, -(1:2)], 2, function(x){ifelse(x > as.numeric(quantile(x)[4]), 'high', 'low')})
attach(dat)
my.surv <- Surv(dat$OS_MONTHS, dat$OS_STATUS == '1:DECEASED')

kmfit <- survfit(my.surv ~ CD3D, data = dat)
p1 <- ggsurvplot(kmfit, conf.int = FALSE, pval = TRUE, risk.table = FALSE,
                 ncensor.plot = FALSE, palette = c('red','blue'))

kmfit <- survfit(my.surv ~ AKT1, data = dat)
p2 <- ggsurvplot(kmfit, conf.int = FALSE, pval = TRUE, risk.table = FALSE,
                 ncensor.plot = FALSE, palette = c('red','blue'))

kmfit <- survfit(my.surv ~ LCK, data = dat)
p3 <- ggsurvplot(kmfit, conf.int = FALSE, pval = TRUE, risk.table = FALSE,
                 ncensor.plot = FALSE, palette = c('red','blue'))

kmfit <- survfit(my.surv ~ TLR4, data = dat)
p4 <- ggsurvplot(kmfit, conf.int = FALSE, pval = TRUE, risk.table = FALSE,
                 ncensor.plot = FALSE, palette = c('red','blue'))

kmfit <- survfit(my.surv ~ STAT1, data = dat)
p5 <- ggsurvplot(kmfit, conf.int = FALSE, pval = TRUE, risk.table = FALSE,
                 ncensor.plot = FALSE, palette = c('red','blue'))

kmfit <- survfit(my.surv ~ STAT3, data = dat)
p6 <- ggsurvplot(kmfit, conf.int = FALSE, pval = TRUE, risk.table = FALSE,
                 ncensor.plot = FALSE, palette = c('red','blue'))

kmfit <- survfit(my.surv ~ UBB, data = dat)
p7 <- ggsurvplot(kmfit, conf.int = FALSE, pval = TRUE, risk.table = FALSE,
                 ncensor.plot = FALSE, palette = c('red','blue'))

kmfit <- survfit(my.surv ~ CD86, data = dat)
p8 <- ggsurvplot(kmfit, conf.int = FALSE, pval = TRUE, risk.table = FALSE,
                 ncensor.plot = FALSE, palette = c('red','blue'))

gh <- cowplot::plot_grid(p1$plot, p2$plot, p3$plot, p4$plot, p5$plot, p6$plot, p7$plot, p8$plot)
ggsave(gh,filename = ,width = 10, height = 7,device = cairo_pdf)
