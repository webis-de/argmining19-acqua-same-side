using NLP.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace NLP
{
    public class ClassificationResult
    {
        private readonly string name;
        private readonly SortedDictionaryWithDuplicates<double, bool> probabilityAndResult = new SortedDictionaryWithDuplicates<double, bool>();
        private Dictionary<int, SortedDictionaryWithDuplicates<double, bool>> probabilityAndResultForTopic = new Dictionary<int, SortedDictionaryWithDuplicates<double, bool>>();

        public ClassificationResult(string name)
        {
            this.name = name;
        }
        
        public void AddPrediction(int topicId, double confidence, bool correct)
        {
            if (!probabilityAndResultForTopic.ContainsKey(topicId))
                probabilityAndResultForTopic.Add(topicId, new SortedDictionaryWithDuplicates<double, bool>());
            probabilityAndResultForTopic[topicId].Add(confidence, correct);

            probabilityAndResult.Add(confidence, correct);
        }

        public double GetMacroAveragedAccuracyFor(ref double coverage)
        {
            int topicCount = probabilityAndResultForTopic.Keys.Count;
            double accuracy = 0;
            double realCoverage = 0;
            foreach (var topicId in probabilityAndResultForTopic.Keys)
            {
                double getCoverage = coverage;
                accuracy += GetAccuracyForTopic(topicId, ref getCoverage);
                realCoverage += getCoverage;
            }

            coverage = realCoverage / topicCount;
            return accuracy / topicCount;
        }

        private double GetAccuracyForTopic(int topicId, ref double coverage)
        {
            int coverageCount = (int)(coverage * probabilityAndResultForTopic[topicId].Count);
            double coverageLimit = GetCoverageLimitForTopic(topicId,coverageCount); //include entries with same confidence


            int count = 0; int correct = 0;
            foreach (var tuple in probabilityAndResultForTopic[topicId])
            {
                if (tuple.Item1 < coverageLimit)
                    break;
                else if (tuple.Item2)
                    correct++;

                count++;
            }

            coverage = count *1.0 / probabilityAndResultForTopic[topicId].Count;

            return correct * 1.0 / count;
        }

        private double GetCoverageLimitForTopic(int topicId, int coverageCount)
        {
            int count = 0;
            foreach (var tuple in probabilityAndResultForTopic[topicId])
            {
                count++;
                if (count > coverageCount)
                {
                    return tuple.Item1;
                }
            }
            return 0;
        }



        public double GetMicroAveragedAccuracyFor(ref double coverage)
        {
            int coverageCount = (int)(coverage * probabilityAndResult.Count);
            double coverageLimit = GetCoverageLimit(coverageCount); //include entries with same confidence


            int count = 0; int correct = 0;
            foreach (var tuple in probabilityAndResult)
            {
                if (tuple.Item1 < coverageLimit)
                    break;
                else if (tuple.Item2)
                    correct++;

                count++;
            }

            coverage = count * 1.0 / probabilityAndResult.Count;

            return correct * 1.0 / count;
        }

        private double GetCoverageLimit(int coverageCount)
        {
            int count = 0;
            foreach (var tuple in probabilityAndResult)
            {
                count++;
                if (count > coverageCount)
                {
                    return tuple.Item1;
                }
            }
            return 0;
        }

        string GetStringFor(double minCoverage, double macroCoverage, double macroAccuracy, double microCoverage, double microAccuracy)
        {
            return new StringBuilder().
                Append("Coverage: ").Append(Format(minCoverage)).
                Append(" -> Macro Averaged: ").Append(Format(macroAccuracy)).Append(" @ ").Append(Format(macroCoverage)).
                Append(", Micro Averaged: ").Append(Format(microAccuracy)).Append(" @ ").AppendLine(Format(microCoverage))
                .ToString();
        }

        private string Format(double value) => string.Format("{0:0.000}", value);

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder(name).AppendLine();//.Append(" -> In total classified: ").AppendLine(probabilityAndResult.Count.ToString());

            for (int i = 10; i <= 100; i += 10)
            {
                double minCoverage = i * 0.01;
                double microCoverage = minCoverage;
                double microAccuracy = GetMicroAveragedAccuracyFor(ref microCoverage);
                double macroCoverage = minCoverage;
                double macroAccuracy = GetMacroAveragedAccuracyFor(ref macroCoverage);
                sb.Append(GetStringFor(minCoverage, macroCoverage, macroAccuracy, microCoverage, microAccuracy));
            }
            return sb.ToString();
        }
    }
}