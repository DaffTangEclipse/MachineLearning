# 决策树生成
class Node(object):
    def __init__(self,root=None,maxcnum=10**4):
        self.root = root
        self.children = []
        self.max_children_num = maxcnum
    def __str__(self):
        return '[' + str(self.root) + ',' + str([each.root for each in self.children]) + ']'

class Tree(object):
    def __init__(self):
        self.root = None
    def add(self,node):
        if self.root is None:
            self.root = node
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            if len(cur_node.children) < cur_node.max_children_num:
                cur_node.children.append(node)
                return
            else:
                for each in cur_node.children:
                    queue.append(each)
    def breadth_travel(self):
        if self.root is None:
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            print(cur_node.root)
            if cur_node.children:
                for each in cur_node.children:
                    queue.append(each)
    def preorder(self,node):
        if node is None:
            return
        print(node.root)
        for each in node.children:
            self.preorder(each)      


def decision_class(data):
    value_list = list(set(data.loc[:,['好瓜']].values.flatten()))
    if len(value_list) >= 2:
        class_dict = {value_list[0]:len(data.loc[data['好瓜']==value_list[0],['好瓜']]),value_list[1]:len(data.loc[data['好瓜']==value_list[1],['好瓜']])}
        dclass = max(class_dict,key=class_dict.get)
        return dclass
    else:
        return value_list[0]            


# 深度优先生成决策树,未剪枝
# divide_fun为划分指标，Gain,Gain_ratio,Gini_index
def TreeGenerate(data,character_set,tree=Tree(),divide_fun=Gain):
    # 生成node 
    dnode = Node(root={'temp':list(data.loc[:,['编号']].values.flatten())},maxcnum=1)
    # D中样本属于同一类别C
    if len(set(data.loc[:,['好瓜']].values.flatten())) == 1: 
        # 将node标记为C类叶结点
        dnode.root[list(set(data.loc[:,['好瓜']].values.flatten()))[0]] = dnode.root.pop('temp')
        tree.add(dnode)
        return 
    # character_set为空 or data中样本在attr上取值相同
    if len(character_set) == 0 or len(set([tuple(each) for each in data.loc[:,list(character_set)].values])) <= 1: 
        # 将node标记为叶结点，其类别标记为D中样本数最多的类
        dnode.root[decision_class(data)] = dnode.root.pop('temp')
        tree.add(dnode)
        return 
    # 从A中选择最优划分属性attr_best
    character_dict = {character:attr for character,attr in zip(character_set,[set(data.loc[:,[each]].values.flatten()) for each in character_set])}
    gain_dict = {character:divide_fun(data,character) for character in character_set} 
    attr_best = max(gain_dict,key=gain_dict.get)
    if divide_fun == Gini_index:
        attr_best = min(gain_dict,key=gain_dict.get)
    anode = Node(root=f'{attr_best}',maxcnum=len(character_dict[attr_best]))
    tree.add(anode)
    # 遍历最优划分属性attr_best的取值
    for attr in character_dict[attr_best]:
        # 为node生成一个分支
        tnode = Node(root={f'{attr_best}={attr}':list(data.loc[data[attr_best]==attr,['编号']].values.flatten())},maxcnum=1)
        # 令D_v表示D在attr上取值为attr_v的样本子集
        attr_index = [i-1 for i in data.loc[data[attr_best]==attr,['编号']].values.flatten()]
        data_v = data.loc[attr_index,:]
        if len(data_v) == 0: # D_v为空
            # 将分支结点标记为叶结点，其类别标记为D中样本最多的类
            tnode.root[decision_class(data)] = tnode.root.pop(f'{attr_best}={attr}')
            tree.add(tnode)
            return 
        else:
            # 以TreeGenerate(D_v,characters\{attr_best})为分支结点
            tree.add(tnode)
            character_set = list(set(character_set) - {attr_best})
            TreeGenerate(data=data_v,character_set=character_set,tree=tree,divide_fun=divide_fun)

character_set0 = list(data0.columns.drop(['编号','好瓜']))
def main(data=data0,character_set=character_set0,divide_fun=Gain):
    tree = Tree()
    TreeGenerate(data,character_set,tree=tree,divide_fun=divide_fun)
    tree.breadth_travel() 
#     tree.preorder(tree.root)
    return tree


# ['密度', '脐部']
# ['脐部', '含糖率', '触感', '敲声', '密度', '色泽', '纹理']
# ['根蒂', '脐部', '含糖率', '触感', '敲声', '密度', '色泽', '纹理']



# 基尼系数
def Gini(data,classi=classi,precision=6):
    Gini = 1
    D = data.loc[:,[classi]]
    for each in set(data.loc[:,[classi]].values.flatten()):
        p = len(data.loc[data[classi]==each,[classi]].values)/len(D.values)
        Gini += -p**2
    return round(Gini,6)

def Gini_index(data,character,classi=classi,precision=precision):
    Gini_a = 0
    for value in set(data.loc[:,[character]].values.flatten()):
        D = data
        D_V = data.loc[data[character]==value,:]
        Gini_a += (len(D_V)/len(D))*Gini(D_V)    
    return round(Gini_a,precision)

# 纹理
# {'纹理=稍糊': [7, 9, 13, 14, 17]}
# 触感
# {'触感=硬滑': [9, 13, 14, 17]}
# {'坏瓜': [9, 13, 14, 17]}
# {'触感=软粘': [7]}
# {'好瓜': [7]}
# {'纹理=模糊': [11, 12, 16]}
# {'坏瓜': [11, 12, 16]}
# {'纹理=清晰': [1, 2, 3, 4, 5, 6, 8, 10, 15]}
# 密度
# {'密度=大于0.381': [1, 2, 3, 4, 5, 6, 8]}
# {'好瓜': [1, 2, 3, 4, 5, 6, 8]}
# {'密度=小于0.381': [10, 15]}
# {'坏瓜': [10, 15]}


# 深度优先生成决策树,预剪枝
# divide_fun为划分指标，Gain,Gain_ratio,Gini_index
def TreeGenerateWithPreCut(data_train,data_test,character_set,tree=Tree(),divide_fun=Gain):
    # 生成node 
    dnode = Node(root={'temp':list(data_train.loc[:,['编号']].values.flatten())})
    # D中样本属于同一类别C
    if len(set(data_train.loc[:,['好瓜']].values.flatten())) <= 1: 
        # 将node标记为C类叶结点
        dnode.root[list(set(data_train.loc[:,['好瓜']].values.flatten()))[0]] = dnode.root.pop('temp')
        tree.add(dnode)
        return 
    # 从A中选择最优划分属性attr_best
    character_dict = {character:attr for character,attr in zip(character_set,[set(data_train.loc[:,[each]].values.flatten()) for each in character_set])}
    gain_dict = {character:divide_fun(data_train,character) for character in character_set} 
    attr_best = max(gain_dict,key=gain_dict.get)
    if divide_fun == Gini_index:
        attr_best = min(gain_dict,key=gain_dict.get)
    # 计算验证集上划分前后的精度accuracy
    dclass = decision_class(data_train)
    pre_class_dict = {f'{attr}':decision_class(data_train.loc[data_train[attr_best]==attr,:]) for attr in character_dict[attr_best]}
    accuracy_pre = len(data_test.loc[data_test.loc[:,'好瓜']==dclass,:])/len(data_test)
    correct_aft = [len(data_test.loc[(data_test[attr_best]==attr)&(data_test['好瓜']==dclass),:]) for attr,dclass in zip(pre_class_dict.keys(),pre_class_dict.values())]
    accuracy_aft = sum(correct_aft)/len(data_test)
    # character_set为空 or data中样本在attr上取值相同 or 验证集划分前精度不小于划分后精度
    if len(character_set) == 0 or len(set([tuple(each) for each in data_train.loc[:,list(character_set)].values])) <= 1 or accuracy_pre >= accuracy_aft: 
        # 将node标记为叶结点，其类别标记为D中样本数最多的类
        dnode.root[decision_class(data_train)] = dnode.root.pop('temp')
        tree.add(dnode)
        return 
    # 选择最优划分属性attr_best为分支结点
    anode = Node(root=f'{attr_best}')
    tree.add(anode)
    # 遍历最优划分属性attr_best的取值
    for attr in character_dict[attr_best]:
        # 为node生成一个分支
        tnode = Node(root={f'{attr_best}={attr}':list(data_train.loc[data_train[attr_best]==attr,['编号']].values.flatten())})
        # 令D_v表示D在attr上取值为attr_v的样本子集
        attr_index = [i-1 for i in data_train.loc[data_train[attr_best]==attr,['编号']].values.flatten()]
        data_v = data_train.loc[attr_index,:]
        if len(data_v) == 0: # D_v为空
            # 将分支结点标记为叶结点，其类别标记为D中样本最多的类
            tnode.root[decision_class(data_train)] = tnode.root.pop(f'{attr_best}={attr}')
            tree.add(tnode)
            return 
        else:
            # 以TreeGenerate(D_v,characters\{attr})为分支结点
            tree.add(tnode)
            character_set1 = list(set(character_set) - {attr_best})
            TreeGenerateWithPreCut(data_train=data_v,data_test=data_test,character_set=character_set1,tree=tree,divide_fun=divide_fun)

# data1
train_index = [1,2,3,6,7,10,14,15,16,17]
test_index = [4,5,8,9,11,12,13]
data_train = data1.loc[[i-1 for i in train_index],:]
data_test = data1.loc[[i-1 for i in test_index],:]
character_set1 = list(data1.columns.drop(['编号','好瓜']))
def PreCutTree(data_train=data_train,data_test=data_test,character_set=character_set1,divide_fun=Gini_index):
    character_set1 = list(data1.columns.drop(['编号','好瓜']))
    tree = Tree()
    TreeGenerateWithPreCut(data_train=data_train,data_test=data_test,character_set=character_set1,tree=tree,divide_fun=divide_fun)
    tree.breadth_travel() 
    return tree

# 预减枝与未剪枝
divide = Gain
PreCutTree(data_train=data_train,data_test=data_test,divide_fun=divide)
print()
main(data=data1,character_set=character_set1,divide_fun=Gain_ratio)

# 色泽
# {'色泽=乌黑': [2, 3, 7, 15]}
# {'好瓜': [2, 3, 7, 15]}
# {'色泽=浅白': [14, 16]}
# {'坏瓜': [14, 16]}
# {'色泽=青绿': [1, 6, 10, 17]}
# 敲声
# {'敲声=清脆': [10]}
# {'坏瓜': [10]}
# {'敲声=沉闷': [17]}
# {'坏瓜': [17]}
# {'敲声=浊响': [1, 6]}
# {'好瓜': [1, 6]}



# 纹理
# {'纹理=稍糊': [7, 9, 13, 14, 17]}
# 触感
# {'触感=硬滑': [9, 13, 14, 17]}
# {'坏瓜': [9, 13, 14, 17]}
# {'触感=软粘': [7]}
# {'好瓜': [7]}
# {'纹理=模糊': [11, 12, 16]}
# {'坏瓜': [11, 12, 16]}
# {'纹理=清晰': [1, 2, 3, 4, 5, 6, 8, 10, 15]}
# 触感
# {'触感=硬滑': [1, 2, 3, 4, 5, 8]}
# {'好瓜': [1, 2, 3, 4, 5, 8]}
# {'触感=软粘': [6, 10, 15]}
# 根蒂
# {'根蒂=稍蜷': [6, 15]}
# 色泽
# {'色泽=乌黑': [15]}
# {'坏瓜': [15]}
# {'色泽=青绿': [6]}
# {'好瓜': [6]}
# {'根蒂=硬挺': [10]}
# {'坏瓜': [10]}