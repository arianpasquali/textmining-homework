require(tm)
library(wordcloud)
library(stringr)
# library(Snowball)

args <- commandArgs()
dataset_dir <- args[4]
use_idf <- as.logical(args[5])
k_size <- as.integer(args[6])
min_freq <- as.integer(args[7])
max_words <- as.integer(args[8])

if(is.na(min_freq)){
  min_freq = 15;
}

if(is.na(max_words)){
  max_words = 100;
}

if(is.na(use_idf)){
  use_idf = TRUE;
}

if(is.na(dataset_dir)){
  stop("Please, inform directory with the dataset. \n Usage build_wordclouds.r [path] [use_idf] [k clusters] [min term freq] [max words]")
}

get_dataset_name <- function(directory){
  print(directory)
  return(basename(directory))
}

load_categories <- function(path=".", pattern=NULL, all.dirs=FALSE,
  full.names=FALSE, ignore.case=FALSE) {

  all <- list.files(path, pattern, all.dirs,
           full.names, recursive=FALSE, ignore.case)
  
  print (all)
  return (all)
}

clean_corpus<- function(corpus){
  corpus <- tm_map(corpus, function(x) str_replace_all(x, "[[:punct:]]", " "))

  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, removePunctuation)  
  corpus <- tm_map(corpus, removeNumbers)

  
  return(corpus) 
}


load_dataset <- function(categories, directory) {
  dirs <- sprintf("%s/%s", directory, categories)
  corpus <- Corpus(DirSource(directory = dirs, encoding = "ANSII"))
  return (corpus)
}

export_wordcloud <- function(term_freq,cluster_id,dataset_name){
  # filename <- paste(dataset_name,"/",sep="")
  # filename <- paste(filename,cluster_id,sep="")
  filename <- paste(cluster_id,"_",sep="")
  filename <- paste(filename,dataset_name,sep="")
  filename <- paste(filename,".png",sep="")
  
  png(file = filename, bg = "white")
  wordcloud(names(term_freq),term_freq,min.freq=min_freq,max.words=max_words)
  dev.off()
}

categories <- load_categories(path=dataset_dir)

# if you didn't inform the number of clusters it will 
# do dirty trick to get it.
cluster_size = length(categories)

if(!is.na(k_size)){
  cluster_size = k_size
}

corpus <- load_dataset(categories,dataset_dir)
corpus <- clean_corpus(corpus)

if(!use_idf){
  doc_term_matrix<-DocumentTermMatrix(corpus)
}else{
  doc_term_matrix<-weightTfIdf(DocumentTermMatrix(corpus))
}

doc_term_matrix <- removeSparseTerms(doc_term_matrix,0.9)
# distance_matrix <- dist(doc_term_matrix);

#generate clusters with hierachical method
# tree <- hclust(dissimilarity(doc_term_matrix,method="cosine"),method="ward")
tree <- hclust(dist(doc_term_matrix),method="ward")
cluster_key<-cutree(tree,cluster_size)

# plot wordclouds
dataset_name = get_dataset_name(dataset_dir)
for(i in 1:cluster_size) {
    cluster <- doc_term_matrix[cluster_key==i,]
    term_frequency<-apply(cluster,2,sum)
    term_frequency[order(term_frequency,decreasing=T)[1:100]]
    export_wordcloud(term_frequency,i,dataset_name);    
}