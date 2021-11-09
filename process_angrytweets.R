# LIBRARIES
pacman::p_load(tidyverse, rtweet)
# FUNCTIONS #
extract_annotations <- function(str_vec) {
  # Extract all the words
  extracted_words <- str_extract_all(str_vec, "\\w+")
  # Figure out whether all words are the same
  is_the_same <- map_lgl(extracted_words, ~all(.x ==.x[1]))
  first_word <- map_chr(extracted_words, 1) # First word from sublist
  return(ifelse(is_the_same, first_word, NA))
}
angry_raw <- read_csv("./data/game_tweets.csv", 
                      col_types = cols(twitterid = col_character()))



angry_clean <- angry_raw %>% 
  mutate(annotation = extract_annotations(annotation), 
         annotation = na_if(annotation, "skip")) %>% 
  drop_na()



angry_scraped <- lookup_statuses(angry_raw$twitterid)
angry_scraped_clean <- angry_scraped %>% 
  select(status_id, created_at, screen_name, text, favorite_count, retweet_count)

angry_join <- angry_clean %>% 
  left_join(angry_scraped_clean, by = c("twitterid"="status_id"))
