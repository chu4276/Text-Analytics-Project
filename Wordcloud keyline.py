# tagset_allowed = ['NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ','JJ','JJR','JJS']
import pandas as pd
keyline = pd.read_csv('df_keyline_output.csv')
wordcloud = WordCloud(width = 1600, height = 800, max_words = 200, background_color = 'white').generate(text)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.savefig('keyline_wordcloud_nnadj.png')
plt.show()