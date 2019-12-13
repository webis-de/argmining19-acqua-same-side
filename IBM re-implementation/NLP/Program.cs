using DataAccess;
using edu.stanford.nlp.tagger.maxent;
using NLP.StanceClassification;
using NLP.Stemmer;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NLP
{
    class Program
    {
        //first arg -> path to dataset, second arg -> 0: create model file, 1: test, 2: test with creating caching file
        static void Main(string[] args)
        {
            IStemmer stemmer = IStemmer.GetPorter2Stemmer();
            string dataSetPath = args[0];
            int mode = int.Parse(args[1]);
            string word2VecPath = args[2];

            if (mode == 0)
                Train(new DataSet(dataSetPath, stemmer), word2VecPath);
            else if (mode == 1)
                Test(new DataSet(dataSetPath, stemmer), stemmer, word2VecPath, false);
            else if (mode == 2)
                Test(new DataSet(dataSetPath, stemmer), stemmer, word2VecPath, true);
            else
                throw new ArgumentException("Unknown second parameter");
        }

        private static void Test(DataSet dataSet, IStemmer stemmer, string word2VecPath, bool cacheResults)
        {
            var progress = new ProgressReporting(dataSet.TestData.Count);

            var sentimentClassifier = new ClaimSentimentIdentificator(stemmer);
            var targetIdentificator = new TargetIdentificator(word2VecPath);
            var contrastClassifier = new ContrastClassifier();
            var targetResult = new ClassificationResult("Target Identification");
            var simpleSentimentResult = new ClassificationResult("Simple Sentiment");
            var simpleSentimentWithoutContrastResult = new ClassificationResult("Simple Sentiment without Contrast");
            var sentimentResult = new ClassificationResult("Sentiment");
            var sentimentWithoutContrastResult = new ClassificationResult("Sentiment without Contrast");
            var sentimentAndConstrastResult = new ClassificationResult("Sentiment and Contrast");
            var sentimentWithKnownTargetResult = new ClassificationResult("Sentiment with known target");
            var sentimentAndConstrastWithKnownTargetResult = new ClassificationResult("Sentiment and Contrast with known Target");

            foreach (var entry in dataSet.TestData)
            {
                //target identificator
                var target = targetIdentificator.GetNounPhrase(entry.claim, entry.target);
                targetResult.AddPrediction(entry.topicId, 1, IsSignificantlyOverlapping(target.PhraseArray, entry.claimTarget));

                //simple sentiment analyzer
                double simpleSentiment = sentimentClassifier.GetUnweightedSentiment(entry);
                double confidence = Math.Abs(simpleSentiment);
                bool isCorrect = entry.IsPredictionCorrect(simpleSentiment);
                simpleSentimentResult.AddPrediction(entry.topicId, confidence, isCorrect);

                //simple sentiment (ignoring contrast)
                isCorrect = entry.IsPredictionCorrect(simpleSentiment * entry.targetsRelation);
                simpleSentimentWithoutContrastResult.AddPrediction(entry.topicId, confidence, isCorrect);

                //targeted sentiment analyzer
                double sentiment = sentimentClassifier.GetTargetedSentiment(entry, target.PhraseArray);
                confidence = Math.Abs(sentiment);
                isCorrect = entry.IsPredictionCorrect(sentiment);
                sentimentResult.AddPrediction(entry.topicId, confidence, isCorrect);

                //targeted sentiment analzyer + known contrast
                confidence *= contrastClassifier.GetConsistency(entry);
                sentimentAndConstrastResult.AddPrediction(entry.topicId, confidence, isCorrect);

                //targeted sentiment (ignoring contrast)
                isCorrect = entry.IsPredictionCorrect(sentiment * entry.targetsRelation);
                sentimentWithoutContrastResult.AddPrediction(entry.topicId, confidence, isCorrect);

                //targeted sentiment analyzer but using known target
                sentiment = sentimentClassifier.GetTargetedSentiment(entry, entry.claimTarget);
                confidence = Math.Abs(sentiment);
                isCorrect = entry.IsPredictionCorrect(sentiment);
                sentimentWithKnownTargetResult.AddPrediction(entry.topicId, confidence, isCorrect);

                //targeted sentiment analyzer but using known target + known contrast
                confidence *= contrastClassifier.GetConsistency(entry);
                sentimentAndConstrastWithKnownTargetResult.AddPrediction(entry.topicId, confidence, isCorrect);

                progress.Increment();
            }

            if (cacheResults)
                targetIdentificator.SaveCache();

            //unigrams svm
            OutputBaselineResults(dataSet, 3);

            Console.WriteLine(targetResult);
            Console.WriteLine(simpleSentimentResult);
            Console.WriteLine(simpleSentimentWithoutContrastResult);
            Console.WriteLine(sentimentResult);
            Console.WriteLine(sentimentAndConstrastResult);
            Console.WriteLine(sentimentWithoutContrastResult);

            Console.WriteLine(sentimentWithKnownTargetResult);
            Console.WriteLine(sentimentAndConstrastWithKnownTargetResult);

            Console.ReadLine();
        }

        private static ClassificationResult OutputBaselineResults(DataSet dataSet, double c)
        {
            SVMClassifier svm = new SVMClassifier(dataSet, c);
            svm.Train();
            var result = svm.Test();
            Console.WriteLine(result);
            return result;
        }

        static void Train(DataSet dataSet, string word2VecPath)
        {
            var progress = new ProgressReporting(dataSet.TrainData.Count);

            using (TextWriter writer = File.CreateText("trainBase.csv"))
            {
                writer.WriteLine("IsTarget,Child,Wiki,Sentiment,Morphological,WordNet,Word2Vec");
                TargetIdentificator targetIdentificator = new TargetIdentificator(word2VecPath);
                foreach (var entry in dataSet.TrainData)
                {
                    var targetCandidates = targetIdentificator.GetAllNounPhrases(entry.claim, entry.target);
                    foreach (var candidate in targetCandidates)
                    {
                        bool isPositiveExample = IsSignificantlyOverlapping(candidate.NounPhrase.PhraseArray, entry.claimTarget);
                        var line = Concat(isPositiveExample, targetIdentificator.GetTrainFeatures(candidate));
                        writer.WriteLine(line);
                    }

                    progress.Increment();
                }

                targetIdentificator.SaveCache();
            }
        }

        static bool IsSignificantlyOverlapping(string[] candidateString, string[] targetString)
        {
            int elemInIntersection = 0;
            foreach (var token in candidateString)
            {
                if (targetString.Contains(token))
                    elemInIntersection++;
            }

            var elemInUnion = candidateString.Length + targetString.Length - elemInIntersection;
            var jaccardCoefficient = elemInIntersection * 1.0 / elemInUnion;
            return jaccardCoefficient > 0.6;
        }

        static string Concat(object label, object[] features)
        {
            char sep = ',';
            string concated = label.ToString() + sep;
            for (int i = 0; i < features.Length; i++)
            {
                concated += features[i].ToString().Replace(',','.');
                if (i < features.Length - 1)
                    concated += sep;
            }
            return concated;
        }
    }
}
