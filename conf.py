
##配置文件
conf = {

	#数据类型，tabular, image
	"data_type" : "tabular",

	#选择模型mlp,simple-cnn,vgg
	"model_name" : "mlp",

	#处理方法:fed_ccvr
	"no-iid": "",

	#全局epoch
	"global_epochs" : 100,

	#本地epoch
	"local_epochs" : 5,

	#狄利克雷参数
	"beta" : 0.05,

	"batch_size" : 32,

	"weight_decay":1e-5,

    #学习速率
	"lr" : 0.001,

	"momentum" : 0.9,

	#分类
	"num_classes": 5,

	#节点数
	"num_parties":5,

    #模型聚合权值
	"is_init_avg": True,

    #本地验证集划分比例
	"split_ratio": 0.2,

    #标签列名
	"label_column": "label",

	#数据列名
	"data_column": "file",

    #测试数据
# 	"test_dataset": "./data/intrusion/intrusion_test.csv",
    "test_dataset": "../实验1/dp-fedavg-gan/clinical/clinical_test.csv",

    #训练数据
# 	"train_dataset" : "./data/intrusion/intrusion_train.csv",
    "train_dataset": "../实验1/dp-fedavg-gan/clinical/clinical_train.csv",

    #模型保存目录
	"model_dir":"./save_model/",

    #模型文件名
	"model_file":"model.pth",
    
    "discrete_columns": {
        "adult":[
            'workclass',
            'education',
            'marital_status',
            'occupation',
            'relationship',
            'race',
            'gender',
            'native_country'
        ],
        "intrusion":['protocol_type', 'service', 'flag'],
        "credit":[],
        "covtype":
            ['Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type0', 'Soil_Type1',
             'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
             'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 
             'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 
             'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 
             'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 
             'Soil_Type38', 'Soil_Type39'],
    "clinical":["anaemia","diabetes","high_blood_pressure","sex","smoking"]
}
}