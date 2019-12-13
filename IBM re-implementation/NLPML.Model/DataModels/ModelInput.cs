//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using Microsoft.ML.Data;

namespace NLPML.Model.DataModels
{
    public class ModelInput
    {
        [ColumnName("IsTarget"), LoadColumn(0)]
        public bool IsTarget { get; set; }


        [ColumnName("Child"), LoadColumn(1)]
        public bool Child { get; set; }


        [ColumnName("Wiki"), LoadColumn(2)]
        public bool Wiki { get; set; }


        [ColumnName("Sentiment"), LoadColumn(3)]
        public bool Sentiment { get; set; }


        [ColumnName("Morphological"), LoadColumn(4)]
        public float Morphological { get; set; }


        [ColumnName("WordNet"), LoadColumn(5)]
        public float WordNet { get; set; }


        [ColumnName("Word2Vec"), LoadColumn(6)]
        public float Word2Vec { get; set; }


    }
}
