using libsvm;
using System.Linq;

namespace NLP
{
    class SVMClassifier
    {
        readonly double c;
        readonly DataSet dataSet;
        readonly SVMProblemBuilder problemBuilder;
        C_SVC model;

        public SVMClassifier(DataSet dataSet, double c)
        {
            this.dataSet = dataSet;
            this.c = c;
            problemBuilder = new SVMProblemBuilder();
        }

        public void Train()
        {
            var problem = problemBuilder.CreateProblem(dataSet.TrainData, dataSet.Vocabulary);
            model = new C_SVC(problem, KernelHelper.LinearKernel(), c, probability:true);
        }

        public ClassificationResult Test()
        {
            var result = new ClassificationResult("Test");

            foreach (var entry in dataSet.TestData)
            {
                var newX = SVMProblemBuilder.CreateNode(entry.stemmedClaim, dataSet.Vocabulary);
                var predictedSentiment = model.Predict(newX);
                var probability = model.PredictProbabilities(newX);

                result.AddPrediction(entry.topicId, probability.Max(pair => pair.Value),entry.IsPredictionCorrect(predictedSentiment));
            }
            return result;
        }
    }
}
