# Title     : LDA_MODEL
# Objective : Analyze presidential speeches by LDA.
# Created by: willbe
# Created on: 2/12/18

install.packages("stringr")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("tidytext")
install.packages("tm")
install.packages("tidyr")
install.packages("topicmodels")
install.packages("scales")

library("stringr")
library("ggplot2")
library("dplyr")
library("tidytext")
library("tm")
library("tidyr")
library("topicmodels")
library("scales")

# load files
# get listing of .txt files in directory
filenames <- list.files("assignment_2/data/speech",pattern="*.txt")

# Load a csv file.
# Session > Set Working Directory > Choose Directory...
# Set the root folder of project to a working directory.
speeches = read.csv("assignment_2/data/speech.csv", header = T, stringsAsFactors = FALSE)

# Unite date and president to speech
by_speech <- speeches %>%
  unite(speech, date, president)

# Script to word tokens
by_speech_word <- by_speech %>%
  unnest_tokens(word, script)

# Delete stop_words and count words.
speech_word_counts <- by_speech_word %>%
  anti_join(stop_words) %>%
  count(speech, word, sort = TRUE) %>%
  ungroup()

# Make speech_word_counts DTM
speech_dtm <- speech_word_counts %>%
  cast_dtm(speech, word, n)

# Do LDA
topic_num = 27
speech_lda <- LDA(speech_dtm, k = topic_num, control = list(seed = 1234))

# Tidy data is a standard way of mapping the meaning of a dataset to its structure.
# A dataset is messy or tidy depending on how rows,
# columns and tables are matched up with observations, variables and types.
speech_topics_by_beta <- tidy(speech_lda, matrix = "beta")

# Get top 20 words per topic.
speech_top_terms <- speech_topics_by_beta %>%
  group_by(topic) %>%
  top_n(20, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Save plot
png(filename = "top_20_words_per_topic.png", width = 1920, height = 1080)
speech_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()
dev.off()

speech_topics_by_gamma <- tidy(speech_lda, matrix = "gamma")
write.csv(speech_topics_by_gamma, file = "speech_topics_by_gamma.csv")

speech_topics_by_gamma <- speech_topics_by_gamma %>%
  separate(document, c("date", "president"), sep = "_", convert = TRUE)

# reorder presidents in order of topic 1, topic 2, etc before plotting
png(filename = "speech_topics_by_gamma.png", width = 1920, height = 1080)
speech_topics_by_gamma %>%
  mutate(president = reorder(president, gamma * topic)) %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_boxplot() +
  facet_wrap(~ president)
dev.off()

# Select the top topic per speech.
speech_classifications <- speech_topics_by_gamma %>%
  group_by(date, president) %>%
  top_n(1, gamma) %>%
  ungroup()
write.csv(speech_classifications, file = "speech_classifications.csv")

president_topics <- speech_classifications %>%
  count(president, topic) %>%
  group_by(president) %>%
  top_n(1, n) %>%
  ungroup() %>%
  transmute(consensus = president, topic)

president_classifications <- speech_classifications %>%
  inner_join(president_topics, by = "topic") %>%
  filter(president != consensus)

assignments <- augment(speech_lda, data = speech_dtm)

assignments <- assignments %>%
  separate(document, c("date", "president"), sep = "_", convert = TRUE) %>%
  inner_join(president_topics, by = c(".topic" = "topic"))

png(filename = "president_similarity.png", width = 1920, height = 1080)
assignments %>%
  count(president, consensus, wt = count) %>%
  group_by(president) %>%
  mutate(percent = n / sum(n)) %>%
  ggplot(aes(consensus, president, fill = percent)) +
  geom_tile() +
  scale_fill_gradient2(high = "red", label = percent_format()) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1),
        panel.grid = element_blank()) +
  labs(x = "Words were assigned to",
       y = "Words came from",
       fill = "% of assignments")
dev.off()

president_similarity <- assignments %>%
  count(president, consensus, wt = count) %>%
  group_by(president) %>%
  mutate(percent = n / sum(n))
write.csv(president_similarity, file = "president_similarity.csv")
