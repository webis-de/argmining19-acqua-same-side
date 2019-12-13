using edu.stanford.nlp.ling;
using edu.stanford.nlp.trees;
using java.util;
using LinqToWiki.Generated;
using NLP.Data;
using NLP.StanceClassification.Sentiment;
using NLP.Stemmer;
using Syn.WordNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using Word2vec.Tools;

namespace NLP.StanceClassification.Target
{
    public class TargetCandidate
    {
        readonly SentimentLexicon sentimentLexicon;
        public NounPhrase NounPhrase { private set; get; }

        private readonly CachingFile caching;

        public TargetCandidate(NounPhrase nounPhrase, CachingFile caching)
        {
            NounPhrase = nounPhrase;
            this.caching = caching;
            sentimentLexicon = new SentimentLexicon();
        }

        public bool IsChildOfRoot()
        {
            IndexedWord root = GetRoot();

            for (var iter = NounPhrase.Dependencies.iterator(); iter.hasNext();)
            {
                TypedDependency var = (TypedDependency)iter.next();

                if (var.gov().Equals(root) && NounPhrase.PhraseArray.Contains(var.dep().word()))
                {
                    return true;
                }
            }

            return false;
        }

        //was not implemented
        //public int GetDistanceFromChunkLimit()
        //{
        //    return 0;
        //}

        private IndexedWord GetRoot()
        {
            for (var iter = NounPhrase.Dependencies.iterator(); iter.hasNext();)
            {
                TypedDependency var = (TypedDependency)iter.next();

                if (var.reln().getShortName() == "root")
                {
                    return var.dep();
                }
            }

            return null;
        }

        public bool IsWikipediaTitle(Wiki wiki)
        {
            const string methodKey = "Wiki";
            bool? cachedValue = caching.Get<bool?>(methodKey, NounPhrase.PhraseString);

            if (cachedValue != null)
                return (bool)cachedValue;

            bool retValue = false;            
            var result = from s in wiki.Query.search("intitle:\"" + NounPhrase.PhraseString + "\"") select new { s.title };
            try
            {
                var list = result.ToList();
                retValue = list.Count > 0;
            }
            catch (XmlException)
            {
            }

            caching.Add(methodKey, NounPhrase.PhraseString, retValue);

            return retValue;
        }

        public bool IsConnectedToSentiment()
        {
            for (var iter = NounPhrase.Dependencies.iterator(); iter.hasNext();)
            {
                TypedDependency var = (TypedDependency)iter.next();

                var dep = var.dep();
                var gov = var.gov();

                if ((NounPhrase.PhraseArray.Contains(dep.word()) && sentimentLexicon.GetSentiment(gov.word()) != 0) || 
                    (NounPhrase.PhraseArray.Contains(gov.word()) && sentimentLexicon.GetSentiment(dep.word()) != 0))
                {
                    return true;
                }
            }

            return false;
        }


        //not completely sure how it was meant
        public double IsMorphologicalSimiliar(IStemmer lemmatizer)
        {
            var countOfSameStems = 0;
            var stemmedTarget = NounPhrase.TargetArray.Select(word => lemmatizer.Stem(word));

            foreach (var word in NounPhrase.PhraseArray)
            {
                if (stemmedTarget.Contains(lemmatizer.Stem(word)))
                {
                    countOfSameStems++;
                    continue;
                }
            }

            return countOfSameStems * 1.0 / NounPhrase.PhraseArray.Length;
        }

        public float GetWordNetPathLength(WordNetEngine wordNet)
        {
            const string methodKey = "WordNet";
            string valueKey = NounPhrase.PhraseString + " | " + NounPhrase.TargetString;

            float? cachedValue = caching.Get<float?>(methodKey, valueKey);

            if (cachedValue != null)
                return (float)cachedValue;
            else
            {
                var value = wordNet.GetSentenceSimilarity(NounPhrase.PhraseString, NounPhrase.TargetString);
                caching.Add(methodKey, valueKey, value);
                return value;
            }
        }

        public double GetWord2VecCosineSimilarity(Vocabulary vocabulary)
        {
            var phraseRepresentation = vocabulary.GetSummRepresentationOrNullForPhrase(NounPhrase.PhraseString);
            var targetRepresentation = vocabulary.GetSummRepresentationOrNullForPhrase(NounPhrase.TargetArray);

            if (phraseRepresentation == null || targetRepresentation == null)
                return 0;
            else
                return phraseRepresentation.GetCosineDistanceTo(targetRepresentation).DistanceValue;
        }
    }
}
