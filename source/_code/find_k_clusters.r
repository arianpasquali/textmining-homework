require(tm)
library(wordcloud)
library(stringr)
library(cluster)

args <- commandArgs()
dataset_dir <- args[4]
use_idf <- as.logical(args[5])
max_clusters <- as.integer(args[6])

if(is.na(use_idf)){
  use_idf = TRUE;
}

if(is.na(max_clusters)){
  max_clusters = 6;
}



if(is.na(dataset_dir)){
  stop("Please, inform directory with the dataset. \n Usage find_k_clusters.r [path] [use_idf] [max k clusters]")
}

get_dataset_name <- function(directory){
  print(directory)
  return(basename(directory))
}

load_categories <- function(path=".", pattern=NULL, all.dirs=FALSE,
  full.names=FALSE, ignore.case=FALSE) {

  all <- list.files(path, pattern, all.dirs,
           full.names, recursive=FALSE, ignore.case)
  
  return (all)
}

clean_corpus<- function(corpus){
  corpus <- tm_map(corpus, function(x) str_replace_all(x, "[[:punct:]]", " "))
  corpus <- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus,removePunctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeNumbers)
  return(corpus) 
}

load_dataset <- function(categories, directory) {
  dirs <- sprintf("%s/%s", directory, categories)
  corpus <- Corpus(DirSource(directory = dirs, encoding = "ANSII"))
  return (corpus)
}

export_silhouette <- function(avgS,dataset_name){
	filename <- paste("silhouette","_",sep="")
	filename <- paste(filename,dataset_name,sep="")
	filename <- paste(filename,".png",sep="")

	png(file = filename, bg = "white")
	plot(2:length(avgS),avgS,type='b',
	    main='Average Silhouette Coefficient',
	    xlab='nr. clusters')
	dev.off()
}

categories <- load_categories(path=dataset_dir)

# if you didn't inform the number of clusters it will 
# do dirty trick to get it. this must not be considered 
cluster_size = length(categories)

corpus <- load_dataset(categories,dataset_dir)
corpus <- clean_corpus(corpus)

if(!use_idf){
  doc_term_matrix<-DocumentTermMatrix(corpus)
}else{
  doc_term_matrix<-weightTfIdf(DocumentTermMatrix(corpus))
}

distance_matrix <- dist(doc_term_matrix);

avgS <- c()
for(k in 2:max_clusters) {
  cl <- kmeans(distance_matrix,centers=k,iter.max=1000)
  s <- silhouette(cl$cluster,distance_matrix)
  avgS <- c(avgS,mean(s[,3]))
}

export_silhouette(avgS,get_dataset_name(dataset_dir))
  
