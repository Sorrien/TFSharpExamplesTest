using System;
using System.Collections.Generic;
using TensorFlow;

namespace TFTest
{
    class Program
    {
        static void Main(string[] args)
        {
            var simpleFloatArray1 = new float[] { 1.0f, 2.0f, 3.0f };
            var simpleFloatArray2 = new float[] { 4.0f, 5.0f, 1.0f };
            var exampleHelper = new ExampleHelper();

            exampleHelper.ConstantAddExample(3.0f, 4.0f);
            exampleHelper.PlaceholderAddExample(3.0f, 4.0f);
            exampleHelper.PlaceholderAddManyExample(simpleFloatArray1, simpleFloatArray2);
            exampleHelper.VariableExample(1.0f, 2.0f);
            exampleHelper.VariableArrayExample(simpleFloatArray1, simpleFloatArray2);
            exampleHelper.LinearModelExample();
            exampleHelper.LinearModelWithLossExample();
            Console.ReadLine();
        }
    }
}
