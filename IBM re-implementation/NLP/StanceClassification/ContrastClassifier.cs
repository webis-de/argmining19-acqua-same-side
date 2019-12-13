using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NLP.StanceClassification
{
    //perfect dummy contrast classifier to simulate reordering based on target contrast
    class ContrastClassifier
    {
        public double GetConsistency(DataEntry entry)
        {
            return (entry.targetsRelation + 1) / 2;
        }
    }
}
