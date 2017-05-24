/**
 * Created by 77 on 2017/5/23.
 */
import java.io.File;
import java.io.IOException;
import java.util.*;

import jxl.Workbook;
import jxl.write.*;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

import weka.classifiers.trees.J48;//J4.8
import weka.classifiers.bayes.NaiveBayes;//NaiveBayes
import weka.classifiers.functions.SMO;//SVM
import weka.classifiers.functions.MultilayerPerceptron;//Neural Network
import weka.classifiers.lazy.IBk;//KNN

import weka.classifiers.meta.Bagging;//Bagging

public class UseWeka {
    private static final String[] arffFileNames={"breast-w.arff","colic.arff","credit-a.arff","credit-g.arff","diabetes.arff",
            "hepatitis.arff","mozilla4.arff","pc1.arff","pc5.arff","waveform-5000.arff"};
    private static final String[] classifiersName={"J48","NaiveBayes","SMO","MultilayerPerceptron","IBk(k=5)",
            "Bagging of J48","Bagging of NaiveBayes","Bagging of SMO","Bagging of MultilayerPerceptron","Bagging of IBk(k=5)",};

    public static void main(String[] args) throws Exception {
        WritableWorkbook wwb = Workbook.createWorkbook(new File("data/result.xls"));
        WritableCellFormat format = new WritableCellFormat();
        format.setWrap(true);//设置自动换行
        WritableSheet[] sheet=new WritableSheet[2];//创建两个sheet
        String[] sheetName={"Accuracy","AUC"};
        for(int j=0;j<2;j++) {
            sheet[j] = wwb.createSheet(sheetName[j], 0);
            sheet[j].setColumnView(0, 17);
            sheet[j].setRowView(0, 800);
            for (int i = 0; i < 10; i++) {
                sheet[j].setRowView(i + 1, 400);
                sheet[j].addCell(new Label(i + 1, 0, classifiersName[i],format));
                sheet[j].setColumnView(i + 1, 10);
                sheet[j].addCell(new Label(0, i + 1, arffFileNames[i]));
            }
        }

        Classifier[] classifiers=getClassifiers();//10个分类算法

        for(int j=0;j<10;j++) {
            System.out.println("/**************************   " +
                    classifiersName[j]+
                    "   **************************/");
            for(int k=0;k<10;k++) {
                File inputFile = new File("data/" + arffFileNames[k]);// 训练文件
                ArffLoader atf = new ArffLoader();

                try {
                    atf.setFile(inputFile);
                    Instances dataSet = atf.getDataSet(); // 读入训练文件
                    dataSet.setClassIndex(dataSet.numAttributes() - 1);
                    int numClasses = dataSet.numClasses();//标记类别个数
                    //System.out.println(instancesTrain.numClasses());

                    System.out.println(classifiersName[j] + " on " + arffFileNames[k] + " :");
                    Evaluation eval = new Evaluation(dataSet);
                    eval.crossValidateModel(classifiers[j], dataSet, 10, new Random(1));
                    double accuracy=(1 - eval.errorRate()) * 100;
                    System.out.printf("accuracy: %.4f%%\n",accuracy );

                    double sumAUC = 0;
                    for (int i = 0; i < numClasses; i++)
                        sumAUC += eval.areaUnderROC(i);
                    double averAUC = sumAUC / numClasses;
                    System.out.printf("AUC(ROC Area): %.3f\n\n", averAUC);

                    //写入excel文件
                    sheet[0].addCell(new Label(j+1,k+1,String.format("%.4f%%",accuracy)));//accuracy
                    sheet[1].addCell(new Label(j+1,k+1,String.format("%.3f",averAUC)));//AUC
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        wwb.write();
        wwb.close();
    }

    private static Classifier[] getClassifiers() throws Exception{
        Classifier[] classifiers=new Classifier[10];
        classifiers[0] = new J48();
        classifiers[1] = new NaiveBayes();
        classifiers[2] = new SMO();
        classifiers[3] = new MultilayerPerceptron();
        classifiers[4] = new IBk();
        classifiers[4].setOptions(new String[]{"-K","5"});
        classifiers[5]=new Bagging();
        classifiers[5].setOptions(new String[]{"-W","weka.classifiers.trees.J48"});
        classifiers[6]=new Bagging();
        classifiers[6].setOptions(new String[]{"-W","weka.classifiers.bayes.NaiveBayes"});
        classifiers[7]=new Bagging();
        classifiers[7].setOptions(new String[]{"-W","weka.classifiers.functions.SMO"});
        classifiers[8]=new Bagging();
        classifiers[8].setOptions(new String[]{"-W","weka.classifiers.functions.MultilayerPerceptron"});
        classifiers[9]=new Bagging();
        classifiers[9].setOptions(new String[]{"-W","weka.classifiers.lazy.IBk","--","-K","5"});

        return classifiers;
    }
}
