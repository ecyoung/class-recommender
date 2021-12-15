"""
    This file contains several approaches to calculate
     f: (tags, preferences_over_tags, other_course_ratings) -> course_rating

     Approach 1:
     uses subscores for three different categories (type of class, topic, kNN ratings)
     and returns a weighted average of these subscores as the predicted course rating

     Approach 2:
     uses a regression model

     Approach 3:
     uses a neural network
"""

import knn_util
from input_parsing import *
import torch.optim as optim
import heapq
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

# Approach 1: subscores

def getTypeOfClassSubscore(type_of_class_tags, user_preferences):
    """
        Get the type_of_class subscore for a course-user pair
        :param: type_of_class_tags: relevant type tags for course
        :param user_id: user preferences over tags
        :return: subscore
        """
    n = len(type_of_class_tags)
    if n == 0:
        return None
    subscore = 0
    for tag in type_of_class_tags:
        subscore += user_preferences[tag] / n
    return subscore


def getTopicSubscore(topic_tags, user_preferences):
    """
        Get the topic subscore for a course-user pair
        :param: topic_tags: relevant topic tags for course
        :param user_id: user preferences over tags
        :return: subscore
        """
    n = len(topic_tags)
    if n == 0:
        return None
    subscore = 0
    for tag in topic_tags:
        subscore += user_preferences[tag] / n
    return subscore


def getKNNSubscore(ratings, course_id, rho):
    """
        Get the rating subscore for a course-user pair
        :param ratings: a dictionary that maps course_id to ratings by knn
        :param course_id: course for which rating is to be predicted based on knn
        :param rho: average rating over all courses
        :return: subscore
        """
    n = len(ratings[course_id])
    average = 0
    for rating in ratings[course_id]:
        average += rating / n
    knn_predicted_rating = (rho + (n * average)) / (1 + n)
    return knn_predicted_rating


def getPredictedCourseRatingsApproach1(courses, users, user_id, comparatorSet, is_validation, r=10, threshold=1,
                                       k = 20, pref_weight=0.2, course_weight=0.8, type_of_class_weight=.1, topic_weight=0.15,
                                       knn_weight=.75):
    """
    Get the r top recommended courses for a user
    :param courses: dictionary that maps course_id to Course object, contains information of all courses
    :param users: dictionary that maps user_id to User object, contains information of all users
    :param user_id: user id of the target user
    :param k: number of nearest neighbors
    :param threshold: commonality threshold
    :param comparatorSet: set of ids of target preferences
    :param r: number of recommendations to output
    :param is_validation: whether the recommendation is for validation purposes or not
    :param pref_weight: weight of preference similarity between users to determine overall similarity
    :param course_weight: weight of course similarity between users to determine overall similarity
    :return: dictionary mapping course_id to predicted_rating for that course id
    """

    # get user's k nearest neighbors
    kNNs = knn_util.knn(users, user_id, threshold, k, comparatorSet, pref_weight, course_weight)

    # construct a set of course_ids of courses rated by kNN for later reference
    courses_rated_by_knn = set([])

    # a dictionary mapping course_id to a list of all ratings by kNN
    ratings = {}
    for neighbor_id in kNNs:
        for (course_id, rating) in users.get(neighbor_id).courseRatings.items():
            courses_rated_by_knn.add(course_id)
            if course_id not in ratings.keys():
                ratings[course_id] = [rating]
            else:
                ratings[course_id].append(rating)

    # get set of all courses
    courses_to_rate = set(courses.keys())

    # courses taken by the target user
    courses_taken = users.get(user_id).courseRatings.keys()

    # if not validation: only rate courses not taken by user
    # else: only rate courses taken by user
    if not is_validation:
        courses_to_rate = courses_to_rate.difference(courses_taken)
    else:
        courses_to_rate = courses_to_rate.intersection(courses_taken)
    # initialize priority queue, which will store (predicted_rating, course_id) pairs
    heap = []

    # for each course, predict the score and add it to the priority queue
    for course in courses_to_rate:

        # get type_of_class subscore
        type_of_class_score = getTypeOfClassSubscore(courses[course].type_of_class_tags, users[user_id].preferences)

        # get topic subscore
        topic_score = getTopicSubscore(courses[course].topic_tags, users[user_id].preferences)

        # get knn_subscore and adjust weights if necessary
        if course in courses_rated_by_knn:
            knn_score = getKNNSubscore(ratings, course, courses[course].average_rating)
        else:
            knn_score = courses[course].average_rating
            knn_weight = 1
            topic_weight = 0
            type_of_class_weight = 0

        # rescale depending on available information
        if type_of_class_score is not None and topic_score is not None:
            score = (
                                type_of_class_weight * type_of_class_score + topic_weight * topic_score + knn_weight * knn_score) / (
                                type_of_class_weight + topic_weight + knn_weight)

        elif type_of_class_score is None and topic_score is not None:
            score = (topic_weight * topic_score + knn_weight * knn_score) / (topic_weight + knn_weight)

        elif type_of_class_score is not None and topic_score is None:
            score = (type_of_class_weight * type_of_class_score + knn_weight * knn_score) / (
                        type_of_class_weight + knn_weight)
        else:
            score = knn_score

        heapq.heappush(heap, (-score, course))

    if r > len(heap):
        r = len(heap)

    recommendations = []
    for i in range(r):
        course = heapq.heappop(heap)
        recommendations.append((course[1], -course[0]))

    # return only top r courses in priority queue
    return recommendations


def getPredictedCourseRatingsApproach1_opt(kNNs, courses, users, user_id, is_validation, type_of_class_weight=.05, topic_weight=0.15, knn_weight=.8):
    """
    This function is for optimization purposes only.
    Rather than recalculating the set of kNNs for each value of k,
    this function takes the set of kNNs as an additional input
    """

    # get user's k nearest neighbors
    # kNNs = knn_util.knn(users, user_id, threshold, k, comparatorSet, pref_weight, course_weight)

    # construct a set of course_ids of courses rated by kNN for later reference
    courses_rated_by_knn = set([])

    # a dictionary mapping course_id to a list of all ratings by kNN
    ratings = {}
    for neighbor_id in kNNs:
        for (course_id, rating) in users.get(neighbor_id).courseRatings.items():
            courses_rated_by_knn.add(course_id)
            if course_id not in ratings.keys():
                ratings[course_id] = [rating]
            else:
                ratings[course_id].append(rating)

    # get set of all courses
    courses_to_rate = set(courses.keys())

    # courses taken by the target user
    courses_taken = set(users.get(user_id).courseRatings.keys())

    # if not validation: only rate courses not taken by user
    # else: only rate courses taken by user
    if not is_validation:
        courses_to_rate = courses_to_rate.difference(courses_taken)
    else:
        courses_to_rate = courses_taken

    # initialize priority queue, which will store (predicted_rating, course_id) pairs
    heap = []

    # for each course, predict the score and add it to the priority queue
    for course in courses_to_rate:

        # get type_of_class subscore
        type_of_class_score = getTypeOfClassSubscore(courses[course].type_of_class_tags, users[user_id].preferences)

        # get topic subscore
        topic_score = getTopicSubscore(courses[course].topic_tags, users[user_id].preferences)

        # get knn_subscore and adjust weights if necessary
        if course in courses_rated_by_knn:
            knn_score = getKNNSubscore(ratings, course, courses[course].average_rating)
        else:
            knn_score = courses[course].average_rating
            knn_weight = 1
            topic_weight = 0
            type_of_class_weight = 0

        # rescale depending on available information
        if type_of_class_score is not None and topic_score is not None:
            score = (
                                type_of_class_weight * type_of_class_score + topic_weight * topic_score + knn_weight * knn_score) / (
                                type_of_class_weight + topic_weight + knn_weight)

        elif type_of_class_score is None and topic_score is not None:
            score = (topic_weight * topic_score + knn_weight * knn_score) / (topic_weight + knn_weight)

        elif type_of_class_score is not None and topic_score is None:
            score = (type_of_class_weight * type_of_class_score + knn_weight * knn_score) / (
                        type_of_class_weight + knn_weight)
        else:
            score = knn_score

        heapq.heappush(heap, (-score, course))

    recommendations = []
    while len(heap) > 0:
        course = heapq.heappop(heap)
        recommendations.append((course[1], -course[0]))

    # return only top r courses in priority queue
    return recommendations


# Approach 2:   uses a per-user regression model

def getRegressionPrediction(course, user, courses):
    """
    For a given user, constructs a linear multivariable regression model that predicts course ratings from tags
    :param course: the Course to predict
    :param user: the User object of interest
    :param courses: a dictionary of all courses
    :return: unnormalized prediction for this course
    """
    x = []
    y = []

    for rated_course in user.courseRatings:
        y.append(user.courseRatings[rated_course])
        x.append(np.array(getTagVector(courses[rated_course])))


    # run least-squares regression
    x = np.array(x)
    y = np.array(y)
    coeffs = np.linalg.lstsq(x, y, rcond=None)[0]

    predictx = np.array(getTagVector(course))

    # plug in target course into regression model
    return np.dot(coeffs, predictx)



def getTagVector(course):
    """
    :param course: Course object to check
    :return: tags binary vector v where v[i] = 1 if course is associated with tag i
    """
    output = []

    for i in range(18):
        if i <= 4:
            if i in course.type_of_class_tags:
                output.append(1)
            else:
                output.append(0)
        elif i >= 12:
            if i in course.topic_tags:
                output.append(1)
            else:
                output.append(0)
    output.append(1)
    return output


def getPredictedCourseRatingsApproach2(courses, users, user_id, r=5):
    """
    :param courses: dictionary that maps course_id to Course object, contains information of all courses
    :param users: dictionary that maps user_id to User object, contains information of all users
    :param user_id: user id of the target user
    @param: r, Int, number of classes to recommend
    return top r (course_id, predicted_rating) pair
    """
    heap = []
    max_prediction = 0 - math.inf
    min_prediction = math.inf
    for course_id, course in courses.items():
        score = getRegressionPrediction(course, users[user_id], courses)
        if score > max_prediction:
            max_prediction = score
        if score < min_prediction:
            min_prediction = score
        heapq.heappush(heap, (-score, course_id))
    recommendations = []
    if r > len(heap):
        r = len(heap)
    for i in range(r):
        course = heapq.heappop(heap)
        # normalize ratings to a 1-5 scale
        normalized = 1 + 4*((-course[0] - min_prediction)/(max_prediction - min_prediction))
        recommendations.append((course[1], normalized))
    return recommendations



# Approach 3:   uses a neural network

def getInputVector(course, user_preferences):
    output = []

    for i in range(18):
        if i <= 4:
            if i in course.type_of_class_tags:
                output.append(user_preferences[i])
            else:
                output.append(0)
        # elif 4 < i < 12:
        #     output.append(user_preferences[i])
        elif i >= 12:
            if i in course.topic_tags:
                output.append(user_preferences[i])
            else:
                output.append(0)
    for i in range(10):
        output.append(course.average_rating)
    return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(21, 20)
        self.fc2 = nn.Linear(20, 38)
        self.fc3 = nn.Linear(38, 1)
        # other options: convolution, normalize in between layers

    def forward(self, x):
        x = F.relu(self.fc1(x)) # try remove relu
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # other options: max_pool

def neuralNets(input_matrix, target_matrix):
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    # array of corresponding course rating
    # all_output = []
    for k in range(10):
        for i in range(len(input_matrix)):
            input = input_matrix[i]
            target = target_matrix[i]
            input = torch.tensor(input, dtype=torch.float)
            # output = float(net(input))
            output = net(input)
            target = torch.tensor(target, dtype=torch.float)
            loss = criterion(target, output)
            loss.backward()  # bug
            optimizer.step()
        # all_output.append(output)
    # target_matrix = torch.tensor(target_matrix, dtype=torch.float)
    # loss = criterion(torch.tensor(all_output, dtype=torch.float), target_matrix)
    # loss.backward()  # bug
    # optimizer.step()
    return net



def getPredictedCourseRatingsApproach3(user, r, trained_net):
    """
    @param: user, User object, the user that we want to make recommendation for
    @param: r, Int, number of classes to recommend
    return top r (course_id, predicted_rating) pair
    """
    heap = []
    for course_id, course in courses.items():
        input = getInputVector(courses[course_id], user.preferences)
        score = float(trained_net(torch.tensor(input, dtype=torch.float)))
        heapq.heappush(heap, (-score, course_id))
    # maxheap = heapq._heapify_max(heap)
    recommendations = []
    if r > len(heap):
        r = len(heap)
    for i in range(r):
        course = heapq.heappop(heap)
        # course = heapq._heappop_max(maxheap)
        recommendations.append((course[1], -course[0]))
    return recommendations

def get_data(users, courses):
    """
    Return two matrix, input for neural network, and target label for neural network
    """
    input_matrix = []
    courseRatings = []
    for user_id, user in users.items():
        for course_id, course_rating in user.courseRatings.items():
            input = getInputVector(courses[course_id], user.preferences)
            input_matrix.append(input)
            courseRatings.append(course_rating)
    return input_matrix, courseRatings

if __name__ == "__main__":
    # parse input data and format them for neural nets
    (users, courses) = parseInputData("survey_responses.csv", "course_tags.csv")
    input_matrix, courseRatings = get_data(users, courses)
    # get the trained neural network
    trained_net = neuralNets(input_matrix, courseRatings)
    # recommend courses for user 0
    new_user = users[0]# a User object
    # recommendation = getPredictedCourseRatingsApproach3(new_user, 100, trained_net)