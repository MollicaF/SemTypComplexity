library(tidyverse)

lzw = function(strings, 
               vocab=c('O','A','N','C','B','X','D','R','T', 'p', 'q', 'r', 's', 'b', 'd',',', '_'),
               maxiters=100000){
  #vocab = c('O','A','N','C','B','X','D','R','T', 'p', 'q', 'r', 's', 'b', 'd',',', '_')
  
  test = strings %>%
    str_replace_all('NA', 'D') %>%
    str_replace_all('NOR', 'R') %>%
    str_replace_all('NC', 'T') %>%
    str_replace_all('\\(', 'b') %>%
    str_replace_all('\\)', 'd') %>%
    paste0(collapse = '_')

  dic = vocab
  output = NULL
  i = 0
  while(nchar(test) > 0 & i < maxiters){
    i = i + 1
    # Find the longest string in dictionary that matches
    for(v in dic[rev(order(nchar(dic), dic))]){
      if(grepl(paste0('^',v), test)){
        # Emit dictionary index
        output = c(output, match(v, dic))
        # Remove from string
        test = str_remove(test, paste0('^', v))
        # Update the dictionary
        dic = c(dic, paste0(v,substr(test, 1, 1)))
        break
      }
    }
  }
  sum(ceiling(log2(1:length(dic)))[output])
}


