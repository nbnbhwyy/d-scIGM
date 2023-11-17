library(scRNAseq)
library(SingleCellExperiment)
library(scDHA)
library(mclust)
library(Rtsne)
library(monocle3)
library(ggsci)
library(matrixStats)
library(igraph)
library(Matrix)
library(caret)
library(grid)
library(gridExtra)
library(cowplot)
library(TSCAN)
library(slingshot)
library(umap)

start.point <- list( yan = 1, goolam = 1, deng = 268)
all.plot <- list()
datasets <- c("goolam", "yan","deng")
cell.states <- c("zygote", "2cell", "4cell", "8cell", "16cell", "blast", "Inf")
for (dataset in datasets) {
  dataset = "yan"
  label_name = paste0('path_to_label')
  label = read.csv(label_name,header = TRUE,row.names="id")
  if (dataset=='goolam')
    y = label$goolam.colData.cell_type1
  if (dataset=='yan')
    y = label$yan.colData.cell_type1
  if (dataset=='deng')
    y = label$deng.colData.cell_type1
  data_name = paste0('path_to_data')
  data = t(t(read.csv(data_name,header = TRUE,row.names="X")))
  data_name = paste0('path_to_cluster') 
  louvain_type = t(t(read.csv(data_name,header = TRUE,row.names="X")))
  res <- scDHA(data = data, seed = 0)
  
 sc = res
 data_name =paste0('path_to_embedding') 
 data = t(t(read.csv(data_name,header = TRUE,row.names="X")))
 Data = data[,c(0:100)]

 data_name =  paste0('path_to_umap') 
 data_umap = t(t(read.csv(data_name,header = TRUE,row.names="X")))

 tmp.list <- lapply(c(1), function(i) Data)
 if(nrow(tmp.list[[1]]) <= 5e3)
 {
   set.seed(0)
   all.res <- sc$all.res
   tmp.list.or <- tmp.list
 }

 t.final <- matrix(ncol = length(tmp.list), nrow = nrow(tmp.list[[1]]))
 counter <- 1

 for (x in tmp.list) {
   data <- x
   n <- nrow(data)
   adj <- 1 - cor(t(data))
   g <- graph_from_adjacency_matrix(adj, weighted = TRUE, mode = "undirected")
   g <- mst(g)

   dis <- distances(g)

   dis[is.infinite(dis)] <- -1

   result <- start.point[[dataset]]

   t <- dis[result,]

   for (cl in unique(louvain_type)) {
     idx <- which(louvain_type== cl)
     tmp <- t[idx]
     tmp.max <- max(tmp)
     tmp.min <- min(tmp)
     tmp <- (quantile(tmp, 0.75) - quantile(tmp, 0.25)) * (tmp - min(tmp)) / (max(tmp) - min(tmp)) + quantile(tmp, 0.25)
     t[idx] <- tmp
   }

   t.final[, counter] <- t
   counter <- counter + 1
 }

  t.final <- rowMeans2(t.final)
  to.plot <- data.frame(y = y, t = t.final)
  level_order <- factor(to.plot$y, level = cell.states[which(cell.states %in% unique(y))])

  print(cor(as.numeric(level_order), t.final))
  
  r2.value <- round(cor(as.numeric(level_order), t.final)^2, digits = 2)
  
  all.plot[[dataset]]$bp$r2 <- r2.value
  all.plot[[dataset]]$bp$level <- level_order
  all.plot[[dataset]]$bp$data <- to.plot
  set.seed(0)
  
  umap <- Rtsne(sc$latent, dims = 2, perplexity= 5, verbose=TRUE, max_iter = 500,theta = 0.5, pca = F, partial_pca = T, num_threads = 30, check_duplicates = F)
  umap_temp <-umap(sc$latent)
  umap$Y <-data_umap
  y.tmp <- y
  
  tmp <- cbind(data.frame(umap$Y),y.tmp)
  
  colnames(tmp) <- c("UMAP1","UMAP2","label")
  
  tmp$label <- factor(tmp$label, level = cell.states[which(cell.states %in% unique(y))])
  
  tmp1 <- tmp[order(t.final) ,]
  
  t <- 1:nrow(tmp1)
  x <- tmp1$UMAP1
  y <- tmp1$UMAP2
  
  w <- rep(1, length(t))
  w[c(1, length(w))] <- 100000
  
  sx <- smooth.spline(t, x, w, df = 15)
  sy <- smooth.spline(t, y, w, df = 15)
  
  tmp1 <- data.frame(cbind(sx$y, sy$y))
  
  colnames(tmp1) <- c("UMAP1","UMAP2")
  
  all.plot[[dataset]]$sc$tmp <- tmp
  all.plot[[dataset]]$sc$tmp1 <- tmp1
  
}

  for (dataset in datasets) {
    dataset= 'yan'
    tmp1 <- all.plot[[dataset]] 
    
    p <- ggplot(tmp1$bp$data, aes(y = tmp1$bp$level, x = t, group = tmp1$bp$level, fill = tmp1$bp$level, color = tmp1$bp$level)) +
      geom_jitter(size = 2) +
      labs(y = "Development Stage", x = "Pseudo Time", fill = NULL) +
      theme_classic() +     theme(legend.position = "none",
                                    axis.title.y = element_blank(),
                                    axis.text.x = element_blank(),
                                  axis.title.x = element_text(size = 18),
                                  axis.text.y = element_text(size = 18))   +
      scale_color_npg() +
      scale_fill_npg() +
      annotate("text", y = length(unique(tmp1$bp$level)), x = max(tmp1$bp$data$t)*0.3,  label = paste0("R2 = ", tmp1$bp$r2),size=8  )  
    
    sc <- ggplot(tmp1$sc$tmp, aes(x=UMAP1, y=UMAP2, color = label)) + geom_point(size = 2) + theme_classic() + theme(legend.position="right") +
      scale_color_npg()
    sc <- sc + labs(x="UMAP1", y = "UMAP2", color = NULL)
    sc <- sc + theme(legend.position = "none",
                     axis.text = element_blank(),
                     axis.title.x = element_text(size = 18),
                     axis.title.y = element_text(size = 18))
    sc <- sc + geom_path(data = tmp1$sc$tmp1,
                         stat = "identity", position = "identity",
                         mapping = aes(x=UMAP1, y=UMAP2, color = levels(tmp1$sc$tmp$label)[1] ), arrow = arrow(length=unit(0.60,"cm"), type = "closed"),
                         show.legend = F)
    
  }