using DataAccess;
using NLP.Stemmer;
using System.Collections.Generic;
using System.Linq;

namespace NLP
{
    class DataSet
    {
        public List<DataEntry> TestData { get; private set; }
        public List<DataEntry> TrainData { get; private set; }
        public List<string> Vocabulary { get; private set; }

        public DataSet(string path, IStemmer stemmer)
        {
            TestData = new List<DataEntry>();
            TrainData = new List<DataEntry>();
            Vocabulary = new List<string>();
            var dataTable = DataTable.New.ReadCsv(path);

            foreach (var row in dataTable.Rows)
            {
                var entry = DataEntry.ParseRow(row, stemmer);
                if (entry!=null)
                {
                    if (entry.isTest)
                    {
                        AddAsTestData(entry);
                    }
                    else
                    {
                        AddAsTrainData(entry);
                    }
                }
            }

            Vocabulary = Vocabulary.Distinct().ToList();
            Vocabulary.Sort();
        }

        private void AddAsTestData(DataEntry entry)
        {
            TestData.Add(entry);
        }

        private void AddAsTrainData(DataEntry entry)
        {
            TrainData.Add(entry);
            Vocabulary.AddRange(entry.stemmedClaim);
        }
    }
}
