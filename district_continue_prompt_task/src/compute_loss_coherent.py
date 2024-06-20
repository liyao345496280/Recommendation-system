import torch
import torch.nn.functional as F
#类定义
class compulate:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    #定义构造方法
    def __init__(self):
        self.warmup_proportion = 0.1
        self.centroid_mode = 'mean'
        self.feature_distance_mode = 'cosine'
        self.feature_distance_lower_bound = 1.0
        self.feature_distance_upper_bound = 0.1
        self.score_distance_lower_bound = 0.3
        self.score_distance_upper_bound = 0.1
        self.weighted_s_loss = True
        self.feature_loss_weight = 1.0
        self.score_loss_weight = 0.0
        self.bce_loss_weight = 0.0
        self.use_projection_head = False


        self.cur_distance_mode = None
        self.cur_upper_bound = None
        self.cur_lower_bound = None

        self.bce_criterion = torch.nn.BCELoss()

    def _compute_dual_mlr_loss(self,all_features):
        """#TODO
        all_features and all_scores must be sorted, inside which the former one
        is better than the latter one.
        """
        self.cur_distance_mode = self.feature_distance_mode
        self.cur_lower_bound = self.feature_distance_lower_bound
        self.cur_upper_bound = self.feature_distance_upper_bound
        '''
        all_features:->list:3,list[0]=(5,3,768)
        '''
        feature_s_loss = self._compute_separation_loss(all_features)
        feature_c_loss = self._compute_compactness_loss(all_features)
        feature_o_loss = self._compute_order_loss(all_features)
        feature_s_loss *= self.feature_loss_weight
        feature_c_loss *= self.feature_loss_weight
        feature_o_loss *= self.feature_loss_weight
        feature_mlr_loss = feature_s_loss + feature_c_loss + feature_o_loss
        self.cur_distance_mode = 'l1'
        self.cur_lower_bound = self.score_distance_lower_bound
        self.cur_upper_bound = self.score_distance_upper_bound
        #score_s_loss = self._compute_separation_loss(all_scores)
        #score_c_loss = self._compute_compactness_loss(all_scores)
        #score_o_loss = self._compute_order_loss(all_scores)
        #score_s_loss *= self.score_loss_weight
        #score_c_loss *= self.score_loss_weight
        #score_o_loss *= self.score_loss_weight
        #score_mlr_loss = score_s_loss + score_c_loss + score_o_loss
        #bce_loss = self._compute_bce_loss(all_scores)
        #bce_loss *= self.bce_loss_weight
        dual_mlr_loss = feature_mlr_loss #+ score_mlr_loss + bce_loss
        # loss_info_dict = {
        #     'dual_mlr_loss': dual_mlr_loss.item(),
        #     'bce_loss': bce_loss.item(),
        #     'feature_mlr_loss': feature_mlr_loss.item(),
        #     'score_mlr_loss': score_mlr_loss.item(),
        #     'feature_s_loss': feature_s_loss.item(),
        #     'score_s_loss': score_s_loss.item(),
        #     'feature_c_loss': feature_c_loss.item(),
        #     'score_c_loss': score_c_loss.item(),
        #     'feature_o_loss': feature_o_loss.item(),
        #     'score_o_loss': score_o_loss.item(),
        # }
        loss_info_dict = {
            'dual_mlr_loss': dual_mlr_loss.item(),
        }
        return dual_mlr_loss, loss_info_dict

    def _compute_separation_loss(self, all_data_points):
        inter_distances, inter_weights = self._compute_inter_cluster_distances(
            all_data_points)
        inter_bounds = self.cur_lower_bound#实际上就是pos离neg1距离为1，pos离neg2距离为2，neg2离neg3距离为1。
        if self.weighted_s_loss:
            inter_bounds *= inter_weights
        separation_loss = F.relu(inter_bounds - inter_distances)
        separation_loss = separation_loss.sum(dim=0).mean()
        return separation_loss
    def _compute_inter_cluster_distances(self,
                                         all_data_points) -> torch.Tensor:
        inter_distance_list = []
        inter_weight_list = []
        centroids = []
        for data_points_in_cur_cluster in all_data_points:
            centroid = self.centroid_function(data_points_in_cur_cluster)
            centroids.append(centroid)#质心；batch_size=3，有3个历史对话，每个历史对话对应了3中不同等级的回复(pos、neg1、neg2)
        for i, better_centroid in enumerate(centroids):
            for j, worse_centroid in enumerate(centroids[i+1:]):
                inter_distance = self.distance_function(
                    better_centroid, worse_centroid)
                inter_distance = inter_distance.unsqueeze(0)
                inter_weight = torch.empty_like(inter_distance)
                inter_weight[0] = j + 1
                inter_distance_list.append(inter_distance)
                inter_weight_list.append(inter_weight)
        inter_distances = torch.cat(inter_distance_list, dim=0)
        inter_weights = torch.cat(inter_weight_list, dim=0)
        return inter_distances, inter_weights
    def _compute_compactness_loss(self, all_data_points):
        intra_distances = self._compute_intra_cluster_distances(all_data_points)
        compactness_loss = F.relu(
            intra_distances - self.cur_upper_bound)
        compactness_loss = compactness_loss.sum(dim=0).mean()
        return compactness_loss
    def _compute_intra_cluster_distances(self,
                                         all_data_points) -> torch.Tensor:
        intra_distance_list = []
        for data_points_in_cur_cluster in all_data_points:
            centroid = self.centroid_function(data_points_in_cur_cluster)#3个不同level的质心向量
            num_data_points = data_points_in_cur_cluster.size(0)
            repeated_centroids = centroid.repeat(
                num_data_points, 1, 1)
            distances = self.distance_function(
                data_points_in_cur_cluster, repeated_centroids)
            intra_distance = distances.mean(dim=0)
            intra_distance = intra_distance.unsqueeze(0)
            intra_distance_list.append(intra_distance)
        intra_distances = torch.cat(intra_distance_list, dim=0)
        return intra_distances
    def _compute_order_loss(self, all_data_points):
        centroids = []
        for data_points_in_cur_cluster in all_data_points:
            centroid = self.centroid_function(data_points_in_cur_cluster)
            centroids.append(centroid)
        order_loss = None
        for i, better_centroid in enumerate(centroids):#better_centroid(3，768)标识pos、neg1、neg2的值
            for worse_centroid in centroids[i+1:]:
                cur_order_loss = F.relu(
                    worse_centroid.norm(dim=-1) - better_centroid.norm(dim=-1))
                if order_loss is None:
                    order_loss = cur_order_loss
                else:
                    order_loss += cur_order_loss
        order_loss = order_loss.mean()
        return order_loss

    def centroid_function(self,features):
        centroid_feature = features.mean(dim=0)
        return centroid_feature

    def distance_function (self,vector1, vector2):
        cosine_distance = 1 - F.cosine_similarity(vector1, vector2, dim=-1)  # vector1（5，3，768）
        return cosine_distance