# Title     : LDA_TUNING
# Objective : Find the best parameter of LDA model.
# Created by: willbe
# Created on: 2/12/18

install.packages("ldatuning")
install.packages("tm")
install.packages("sets")

library("ldatuning")
library("tm")
library("sets")

# Load a csv file.
# Session > Set Working Directory > Choose Directory...
# Set the root folder of project to a working directory.
speeches = read.csv("assignment_2\\data\\speech.csv", header = T)
presidents = as.list(as.set(speeches$president))
best_topic_num = c()
best_arun2010 = c()

for (president in presidents) {
  # select speeches by president:
  speeches_per_president = speeches[speeches$president==president,]

  # pre-processing:
  stop_words <- stopwords("SMART")
  scripts <- gsub("'", "", speeches_per_president$script)  # remove apostrophes
  scripts <- gsub("[[:punct:]]", " ", scripts)  # replace punctuation with space
  scripts <- gsub("[[:cntrl:]]", " ", scripts)  # replace control characters with space
  scripts <- gsub("^[[:space:]]+", "", scripts) # remove whitespace at beginning of documents
  scripts <- gsub("[[:space:]]+$", "", scripts) # remove whitespace at end of documents
  scripts <- tolower(scripts)  # force to lowercase
  
  # tokenize on space and output as a list:
  doc.list <- strsplit(scripts, "[[:space:]]+")
  
  # compute the table of terms:
  term.table <- table(unlist(doc.list))
  term.table <- sort(term.table, decreasing = TRUE)
  
  # remove terms that are stop words or occur fewer than 5 times:
  del <- names(term.table) %in% stop_words | term.table < 5
  term.table <- term.table[!del]
  vocab <- names(term.table)
  
  # now put the documents into the format required by the lda package:
  get.terms <- function(x) {
    index <- match(x, vocab)
    index <- index[!is.na(index)]
    rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
  }
  documents <- lapply(doc.list, get.terms)
  
  # Compute some statistics related to the data set:
  D <- length(documents)  # number of documents (2,000)
  W <- length(vocab)  # number of terms in the vocab (14,568)
  doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
  N <- sum(doc.length)  # total number of tokens in the data (546,827)
  term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus [8939, 5544, 2411, 2410, 2143, ...]
  
  # Make DocumentTermMatrix:
  speeches_dtm = as.DocumentTermMatrix(term.table, weightTf)
  
  # Find the best topic number:
  result <- FindTopicsNumber(
    speeches_dtm,
    topics = seq(from = 2, to = 30, by = 1),
    metrics = c("Arun2010"),
    method = "Gibbs",
    control = list(seed = 77),
    mc.cores = 28L,
    verbose = TRUE
  )
  
  # Plot the result:
  print(president)
  FindTopicsNumber_plot(result)
  min_index = which.min(result$Arun2010)
  best_topic_num = append(best_topic_num, min_index)
  best_arun2010 = append(best_arun2010, result$Arun2010[min_index])
}

final_result = data.frame(best_topic_num, best_arun2010)

