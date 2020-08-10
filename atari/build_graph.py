# import pandas as pd
# import re
# import spacy
from openie import StanfordOpenIE
from collections import Counter

def process_text(text,name):

    top_relations = []
    triples = []

    with StanfordOpenIE() as client:
        print('Text: %s.' % text)
        for triple in client.annotate(text):
            # print('|-', triple)
            top_relations.append(triple['relation'])
            triples.append(triple)

        #terms = set([w for w,_ in Counter(top_relations).most_common(15) if not w in {'is','are','is in','consists of','has','have','controls','is with', 'is represented by','is partially protected by'}])

        for t in triples:
            for term in ["hit","shoot","grab","catch","use","blow","destroy","touch","avoid","collide"]:
                if term in t['relation']: # in terms:
                    print(t)
                    break

        graph_image = name+'_graph.png'
        client.generate_graphviz_graph(text, graph_image)
        print('Graph generated: %s.' % graph_image)

        # with open('corpus/pg6130.txt', 'r', encoding='utf8') as r:
        #     corpus = r.read().replace('\n', ' ').replace('\r', '')

        # triples_corpus = client.annotate(corpus[0:50000])
        # print('Corpus: %s [...].' % corpus[0:80])
        # print('Found %s triples in the corpus.' % len(triples_corpus))
        # for triple in triples_corpus[:3]:
        #     print('|-', triple)


