/**
 * Created by 77 on 2017/5/24.
 */
import java.io.File;
import java.util.*;
import java.util.Map.*;

import jxl.Workbook;
import jxl.write.*;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import weka.classifiers.lazy.IBk;//KNN

public class ABaggingKNN {
    private static final String[] arffFileNames={"breast-w.arff","colic.arff","credit-a.arff","credit-g.arff","diabetes.arff",
            "hepatitis.arff","mozilla4.arff","pc1.arff","pc5.arff","waveform-5000.arff"};

    private static final int folds=10;//10折交叉验证
    private static final int T=100;//基分类器数目

    public static void main(String[] args) throws Exception {
        //写入result_ABaggingOfKNN.xls
        WritableWorkbook wwb = Workbook.createWorkbook(new File("data/result_ABaggingOfKNN.xls"));
        WritableCellFormat format = new WritableCellFormat();
        format.setWrap(true);//设置自动换行
        WritableSheet sheet=wwb.createSheet("ABaggingOfKNN", 0);
        sheet.setColumnView(0, 17);
        sheet.setColumnView(1, 10);
        sheet.setRowView(0, 800);
        sheet.addCell(new Label(1, 0, "ABagging Of IBk(k=5)",format));
        for (int i = 0; i < 10; i++) {
            sheet.setRowView(i + 1, 400);
            sheet.addCell(new Label(0, i + 1, arffFileNames[i]));
        }


        for(int i=0;i<10;i++) {
            File inputFile = new File("data/"+arffFileNames[i]);// 训练文件
            ArffLoader atf = new ArffLoader();
            atf.setFile(inputFile);
            Instances dataSet = atf.getDataSet(); // 读入训练文件
            dataSet.setClassIndex(dataSet.numAttributes()-1);

            // randomize data
            Instances randDataSet = new Instances(dataSet);
            randDataSet.randomize(new Random());
            if (randDataSet.classAttribute().isNominal())
                randDataSet.stratify(folds);
            // perform cross-validation
            double sumAccuracy=0;//累加精确度
            for (int n = 0; n < folds; n++) {//folds
                //划分训练集和测试集
                Instances trainSet = randDataSet.trainCV(folds, n);
                Instances testSet = randDataSet.testCV(folds, n);
                //基分类器
                Classifier[] m_classifiers=new Classifier[T];
                for(int j=0;j<T;j++) {
                    m_classifiers[j] = new IBk();
                    String[] options = {"-K", "5"};
                    m_classifiers[j].setOptions(options);
                }
                //训练集和测试集
                Instances[] trainSets=new Instances[T];
                Instances[] testSets=new Instances[T];
                for(int j=0;j<T;j++) {
                    //对训练集和测试集的属性进行重采样
                    sampleAttributesOfInstances(trainSet,testSet,trainSets,testSets,j);
                    //System.out.println(testSets[j]);
                    m_classifiers[j].buildClassifier(trainSets[j]);//训练第j个基分类器
                }
                //计算ABaggingKNN的accuracy
                int numTestInstances=testSet.numInstances();
                int predictCorrectNum=0;
                for(int k=0;k<numTestInstances;k++){
                    if(vote(m_classifiers,testSets,k)==testSet.instance(k).classValue())
                        predictCorrectNum++;
                }
                double accuracy=((double)predictCorrectNum)/numTestInstances;
                //System.out.println(accuracy);
                sumAccuracy+=accuracy;
            }
            double averageAccuracy=sumAccuracy/folds;
            System.out.println("ABagging of KNN on "+arffFileNames[i]+":");
            System.out.printf("Accuracy: %.2f%%\n\n",averageAccuracy*100);
            sheet.addCell(new Label(1,i+1,String.format("%.4f%%",averageAccuracy*100)));//accuracy
        }

        wwb.write();
        wwb.close();
    }

    //重采样得到新的训练集和测试集
    private static void sampleAttributesOfInstances(Instances trainSet,Instances testSet,Instances[] trainSets,Instances[] testSets,int index){
        trainSets[index]=new Instances(trainSet);
        testSets[index]=new Instances(testSet);
        int numAttributes=trainSets[index].numAttributes();

        //重采样出新的属性列
        int numAttributesLeft=numAttributes*3/4;
        //int numAttributesLeft=numAttributes/2;
        for(int i=0;i<numAttributesLeft;i++){
            int attributeIndex=new Random().nextInt(numAttributes-1);
            //System.out.println(attributeIndex);
            copyAnAttributeOfInstances(trainSets[index],testSets[index],attributeIndex,i);
            //System.out.println(testSets[index]);
        }
        //删除原先的旧属性
        for(int i=0;i<numAttributes-1;i++){
            trainSets[index].deleteAttributeAt(0);
            testSets[index].deleteAttributeAt(0);
        }
        //System.out.println(testSets[index]);
        //更新classIndex
        trainSets[index].setClassIndex(0);
        testSets[index].setClassIndex(0);
    }

    //复制一列属性值
    private static void copyAnAttributeOfInstances(Instances sampleTrainSet,Instances sampleTestSet,int attributeIndex,int index){
        //插入属性名
        Attribute attribute=sampleTrainSet.attribute(attributeIndex);
        int attributeNewIndex=sampleTrainSet.numAttributes();
        sampleTrainSet.insertAttributeAt(attribute.copy(attribute.name()+"_"+index),attributeNewIndex);
        sampleTestSet.insertAttributeAt(attribute.copy(attribute.name()+"_"+index),attributeNewIndex);
        //插入属性值
        for(int i=0;i<sampleTrainSet.numInstances();i++){
            sampleTrainSet.instance(i).setValue(attributeNewIndex,sampleTrainSet.instance(i).value(attributeIndex));
        }
        for(int i=0;i<sampleTestSet.numInstances();i++){
            sampleTestSet.instance(i).setValue(attributeNewIndex,sampleTestSet.instance(i).value(attributeIndex));
        }
    }

    //vote函数
    private static double vote(Classifier[] m_classifiers,Instances[] testSets,int index) throws Exception{
        //int numClasses=testSets[0].numClasses();//标记类别个数
        HashMap<Double,Integer> hashMap=new HashMap<>();
        for(int j=0;j<T;j++){
            double predict=m_classifiers[j].classifyInstance(testSets[j].instance(index));
            if(hashMap.containsKey(predict)){
                int poll=hashMap.get(predict)+1;
                hashMap.replace(predict,poll);
            }
            else{
                hashMap.put(predict,1);
            }
        }
        ArrayList<Entry<Double, Integer>> list = new ArrayList<Entry<Double, Integer>>(hashMap.entrySet());
        Collections.sort(list, new Comparator<Entry<Double, Integer>>() {
            @Override
            public int compare(Entry<Double, Integer> o1, Entry<Double, Integer> o2) {
                if(o1.getValue()<o2.getValue())
                    return -1;
                else if(o1.getValue()>o2.getValue())
                    return 1;
                else
                    return 0;
            }
        });

        return list.get(list.size()-1).getKey();
    }
}
