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
            ConstantAddExample(3.0f, 4.0f);
            PlaceholderAddExample(3.0f, 4.0f);
            PlaceholderAddManyExample(simpleFloatArray1, simpleFloatArray2);
            VariableExample(1.0f, 2.0f);
            VariableArrayExample(simpleFloatArray1, simpleFloatArray2);
            Console.ReadLine();
        }

        public static void ConstantAddExample(float a, float b)
        {
            using (var graph = new TFGraph())
            {
                var node1 = graph.Const(a);
                var node2 = graph.Const(b);
                var node3 = graph.Add(node1, node2);

                using (var session = new TFSession(graph))
                {

                    var output = session.GetRunner().Run(node3);
                    var result = output.GetValue();

                    Console.WriteLine($"ConstantAddExample: {result}");
                }
            }
        }

        public static void PlaceholderAddExample(float a, float b)
        {
            using (var graph = new TFGraph())
            {
                var node1 = graph.Placeholder(TFDataType.Float);
                var node2 = graph.Placeholder(TFDataType.Float);
                var node3 = graph.Add(node1, node2);

                using (var session = new TFSession(graph))
                {
                    var output = session.GetRunner().AddInput(node1, a).AddInput(node2, b).Run(node3);
                    var result = output.GetValue();

                    Console.WriteLine($"PlaceholderAddExample: {result}");
                }
            }
        }
        public static void PlaceholderAddManyExample(float[] a, float[] b)
        {
            using (var graph = new TFGraph())
            {
                var node1 = graph.Placeholder(TFDataType.Float);
                var node2 = graph.Placeholder(TFDataType.Float);
                var node3 = graph.Add(node1, node2);

                using (var session = new TFSession(graph))
                {
                    var output = session.GetRunner().AddInput(node1, a).AddInput(node2, b).Run(node3);
                    var results = (float[])output.GetValue();

                    var printValue = "";
                    foreach (var result in results)
                    {
                        printValue += $"{result},";
                    }
                    Console.WriteLine($"PlaceholderAddManyExample: {printValue}");
                }
            }
        }

        public static void VariableExample(float a, float b)
        {
            using (var graph = new TFGraph())
            {
                var aVar = graph.VariableV2(TFShape.Scalar, TFDataType.Float, operName: "aVar");
                var initA = graph.Assign(aVar, graph.Const(a));

                var bVar = graph.VariableV2(TFShape.Scalar, TFDataType.Float, operName: "bVar");
                var initB = graph.Assign(bVar, graph.Const(b));

                var adderNode = graph.Add(aVar, bVar);

                using (var session = new TFSession(graph))
                {
                    var runInit = session.GetRunner()
                                 .AddTarget(initA.Operation)
                                 .AddTarget(initB.Operation).Run();

                    var output = session.GetRunner().Run(adderNode);
                    var result = (float)output.GetValue();

                    var printValue = result;
                    //foreach (var result in results)
                    //{
                    //    printValue += $"{result},";
                    //}
                    Console.WriteLine($"VariableExample: {printValue}");
                }
            }
        }
        public static void VariableArrayExample(float[] a, float[] b)
        {
            using (var graph = new TFGraph())
            {
                var aVar = graph.VariableV2(TFShape.Scalar, TFDataType.Float, operName: "aVar");
                var initA = graph.Assign(aVar, graph.Const(a));

                var bVar = graph.VariableV2(TFShape.Scalar, TFDataType.Float, operName: "bVar");
                var initB = graph.Assign(bVar, graph.Const(b));

                var adderNode = graph.Add(aVar, bVar);

                using (var session = new TFSession(graph))
                {
                    var runInit = session.GetRunner()
                                 .AddTarget(initA.Operation).Run();
                    var runInit2 = session.GetRunner()
                                                     .AddTarget(initB.Operation).Run();

                    var output = session.GetRunner().Run(adderNode);
                    var results = (float[])output.GetValue();

                    var printValue = "";
                    foreach (var result in results)
                    {
                        printValue += $"{result},";
                    }
                    Console.WriteLine($"VariableArrayExample: {printValue}");
                }
            }
        }
    }
}
