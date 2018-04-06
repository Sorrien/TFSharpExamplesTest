using System;
using System.Collections.Generic;
using TensorFlow;

namespace TFTest
{
    public class ExampleHelper
    {
        public ExampleHelper()
        {

        }

        public void ConstantAddExample(float a, float b)
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

        public void PlaceholderAddExample(float a, float b)
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
        public void PlaceholderAddManyExample(float[] a, float[] b)
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

        public void VariableExample(float a, float b)
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
        public void VariableArrayExample(float[] a, float[] b)
        {
            using (var graph = new TFGraph())
            {
                var aVar = graph.VariableV2(new TFShape(a.Length), TFDataType.Float, operName: "aVar");
                var initA = graph.Assign(aVar, graph.Const(a, TFDataType.Float));

                var bVar = graph.VariableV2(new TFShape(b.Length), TFDataType.Float, operName: "bVar");
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

        public void LinearModelExample()
        {
            using (var graph = new TFGraph())
            {
                var wData = new float[] { 0.3f };
                var bData = new float[] { -0.3f };


                var w = graph.VariableV2(new TFShape(wData.Length), TFDataType.Float, operName: "w");
                var initW = graph.Assign(w, graph.Const(wData, TFDataType.Float));

                var b = graph.VariableV2(new TFShape(bData.Length), TFDataType.Float, operName: "b");
                var initB = graph.Assign(b, graph.Const(bData));

                var x = graph.Placeholder(TFDataType.Float);

                var linearModel = graph.Add(b, graph.Mul(w, x));

                using (var session = new TFSession(graph))
                {
                    var runInit = session.GetRunner()
                                 .AddTarget(initW.Operation).Run();
                    var runInit2 = session.GetRunner()
                                                     .AddTarget(initB.Operation).Run();
                    var output = session.GetRunner()
                        .AddInput(x, new float[] { 1, 2, 3, 4 })
                        .Run(linearModel);
                    var results = (float[])output.GetValue();

                    var printValue = "";
                    foreach (var result in results)
                    {
                        printValue += $"{result},";
                    }
                    Console.WriteLine($"LinearModelExample: {printValue}");
                }
            }
        }

        public void LinearModelWithLossExample()
        {
            using (var graph = new TFGraph())
            {
                var wData = new float[] { 0.3f };
                var bData = new float[] { -0.3f };


                var w = graph.VariableV2(new TFShape(wData.Length), TFDataType.Float, operName: "w");
                var initW = graph.Assign(w, graph.Const(wData, TFDataType.Float));

                var b = graph.VariableV2(new TFShape(bData.Length), TFDataType.Float, operName: "b");
                var initB = graph.Assign(b, graph.Const(bData));

                var x = graph.Placeholder(TFDataType.Float);
                var linearModel = graph.Add(b, graph.Mul(w, x));

                var y = graph.Placeholder(TFDataType.Float);
                var squaredDeltas = graph.SquaredDifference(linearModel, y);
                var loss = graph.ReduceSum(squaredDeltas);



                using (var session = new TFSession(graph))
                {
                    var runInit = session.GetRunner()
                                 .AddTarget(initW.Operation).Run();
                    var runInit2 = session.GetRunner()
                                                     .AddTarget(initB.Operation).Run();
                    var output = session.GetRunner()
                        .AddInput(x, new float[] { 1, 2, 3, 4 })
                        .AddInput(y, new float[] { 0, -1, -2, -3 })
                        .Run(loss);
                    var results = (float)output.GetValue();

                    var printValue = "";
                    //foreach (var result in results)
                    //{
                    //    printValue += $"{result},";
                    //}
                    printValue = $"{results}";
                    Console.WriteLine($"LinearModelWithLossExample: {printValue}");
                }
            }
        }

        public void LinearModelWithTrainingExample()
        {
            using (var graph = new TFGraph())
            {
                var wData = new float[] { 0.3f };
                var bData = new float[] { -0.3f };


                var w = graph.VariableV2(new TFShape(wData.Length), TFDataType.Float, operName: "w");
                var initW = graph.Assign(w, graph.Const(wData, TFDataType.Float));

                var b = graph.VariableV2(new TFShape(bData.Length), TFDataType.Float, operName: "b");
                var initB = graph.Assign(b, graph.Const(bData));

                var x = graph.Placeholder(TFDataType.Float);
                var linearModel = graph.Add(b, graph.Mul(w, x));

                var y = graph.Placeholder(TFDataType.Float);
                var squaredDeltas = graph.SquaredDifference(linearModel, y);
                var loss = graph.ReduceSum(squaredDeltas);

                //var optimizer = graph.ApplyGradientDescent(loss, 0.01, );

                using (var session = new TFSession(graph))
                {
                    var runInit = session.GetRunner()
                                 .AddTarget(initW.Operation).Run();
                    var runInit2 = session.GetRunner()
                                                     .AddTarget(initB.Operation).Run();
                    var output = session.GetRunner()
                        .AddInput(x, new float[] { 1, 2, 3, 4 })
                        .AddInput(y, new float[] { 0, -1, -2, -3 })
                        .Run(loss);
                    var results = (float)output.GetValue();

                    var printValue = "";
                    //foreach (var result in results)
                    //{
                    //    printValue += $"{result},";
                    //}
                    printValue = $"{results}";
                    Console.WriteLine($"LinearModelWithTrainingExample: {printValue}");
                }
            }
        }
    }
}
