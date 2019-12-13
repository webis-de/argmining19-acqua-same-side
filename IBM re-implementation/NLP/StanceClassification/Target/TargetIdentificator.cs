using edu.stanford.nlp.ling;
using edu.stanford.nlp.parser.lexparser;
using edu.stanford.nlp.parser.nndep;
using edu.stanford.nlp.trees;
using LinqToWiki.Generated;
using Microsoft.ML;
using NLP.Data;
using NLP.StanceClassification.Target;
using NLP.Stemmer;
using NLPML.Model.DataModels;
using Syn.WordNet;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Word2vec.Tools;

namespace NLP.StanceClassification
{
    public class TargetIdentificator
    {
        readonly LexicalizedParser lexParser;
        readonly GrammaticalStructureFactory grammaticalStructureFactory;
        readonly IStemmer lemmatizer;
        readonly WordNetEngine wordNetEngine;
        readonly Vocabulary word2VecVocabulary;
        readonly PredictionEngine<ModelInput, ModelOutput> predictionEngine;
        readonly Wiki wiki;
        readonly CachingFile caching;

        public TargetIdentificator(string word2VecModelPath)
        {
            lexParser = LexicalizedParser.loadModel(@"Resources\englishPCFG.ser.gz");
            grammaticalStructureFactory = new PennTreebankLanguagePack().grammaticalStructureFactory();

            lemmatizer = IStemmer.GetLemmatizer();

            wordNetEngine = new WordNetEngine();
            wordNetEngine.LoadFromDirectory(@"Resources\WordNet");

            word2VecVocabulary = new Word2VecBinaryReader().Read(word2VecModelPath);

            MLContext mlContext = new MLContext();
            ITransformer mlModel = mlContext.Model.Load("MLModel.zip", out _);
            predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            wiki = new Wiki("NLP/1.0", "https://en.wikipedia.org", "https://en.wikipedia.org/w/api.php");

            caching = new CachingFile();
        }

        public NounPhrase GetNounPhrase(string[] sentence, string[] target)
        {
            var targetCandidates = GetAllNounPhrases(sentence, target);

            TargetCandidate bestCandidate = null;
            float bestScore = float.NegativeInfinity;

            foreach (var phrase in targetCandidates)
            {
                var input = new ModelInput
                {
                    Child = phrase.IsChildOfRoot(),
                    Morphological = (float)phrase.IsMorphologicalSimiliar(lemmatizer),
                    Sentiment = phrase.IsConnectedToSentiment(),
                    Wiki = phrase.IsWikipediaTitle(wiki),
                    Word2Vec = (float)phrase.GetWord2VecCosineSimilarity(word2VecVocabulary),
                    WordNet = phrase.GetWordNetPathLength(wordNetEngine)
                };

                ModelOutput result = predictionEngine.Predict(input);

                if (result.Score > bestScore)
                {
                    bestCandidate = phrase;
                    bestScore = result.Score;
                }
            }

            return bestCandidate.NounPhrase;
        }

        public List<TargetCandidate> GetAllNounPhrases(string[] sentence, string[] target)
        {
            var tree = lexParser.apply(SentenceUtils.toCoreLabelList(sentence));
            var dependencies = grammaticalStructureFactory.newGrammaticalStructure(tree).typedDependenciesCCprocessed();

            List<TargetCandidate> nounPhrases = new List<TargetCandidate>();

            var subTrees = tree.subTreeList();
            for (int i = 0; i < subTrees.size(); i++)
            {
                Tree subTree = (Tree)subTrees.get(i);
                if (subTree.label().value() == "NP")
                {
                    NounPhrase phrase = NounPhrase.SetSentence(sentence, tree, dependencies, target);
                    phrase.SetPhrase(SentenceUtils.listToString(subTree.yield()));
                    nounPhrases.Add(new TargetCandidate(phrase,caching));
                }
            }

            return nounPhrases;
        }

        public object[] GetTrainFeatures(TargetCandidate candidate)
        {
            return new object[] {
                candidate.IsChildOfRoot(),
                candidate.IsWikipediaTitle(wiki),
                candidate.IsConnectedToSentiment(),
                candidate.IsMorphologicalSimiliar(lemmatizer),
                candidate.GetWordNetPathLength(wordNetEngine),
                candidate.GetWord2VecCosineSimilarity(word2VecVocabulary)
            };
        }

        public void SaveCache()
        {
            caching.SaveToFile();
        }
    }
}
