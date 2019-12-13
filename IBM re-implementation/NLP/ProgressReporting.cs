using System;

namespace NLP
{
    class ProgressReporting
    {
        readonly int wholeCount;
        int index;

        public ProgressReporting(int wholeCount)
        {
            this.wholeCount = wholeCount;
            index = 0;
        }

        public void Increment()
        {
            index++;
            if (index % (wholeCount / 10) == 0)
                Console.WriteLine(Math.Round(index * 100.0 / wholeCount, 1) + " % processed");
        }
    }
}
