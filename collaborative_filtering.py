import heapq
import knn_util

def getPredictedCourseRatings(users, courses, user_id, comparatorSet, is_validation, r = 5, threshold = 1, k=20, pref_weight=0.2, course_weight=0.8):
    """
    Get the r top recommended courses for a user
    :param: users: dictionary that maps user_id to User object, contains information of all users
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
    kNNs = knn_util.knn(users, user_id, threshold, k, comparatorSet, pref_weight, course_weight)
    # a set of course_ids of courses rated by kNN
    rated_courses = set([])
    # a dictionary mapping course_id to a list of all ratings by kNN
    ratings = {}
    for neighbor_id in kNNs:
        # loop through all ratings by this neighbor
        for (course_id, rating) in users.get(neighbor_id).courseRatings.items():
            rated_courses.add(course_id)
            if course_id not in ratings.keys():
                ratings[course_id] = [rating]
            else:
                ratings[course_id].append(rating)

    # courses taken by the target user
    courses_taken = users.get(user_id).courseRatings.keys()

    if not is_validation:
        courses_to_rate = rated_courses.difference(courses_taken)
    else:
        courses_to_rate = rated_courses.intersection(courses_taken)

    # initialize priority queue, which will store (predicted_rating, course_id) pairs
    heap = []

    # for each course, predict the score and add it to the priority queue
    for course in courses_to_rate:
        n = len(ratings[course])
        average = 0
        for rating in ratings[course]:
            average += rating / n
        predicted_rating = (courses[course].average_rating + (n * average)) / (1 + n)
        heapq.heappush(heap, (-predicted_rating, course))

    if r > len(heap):
        r = len(heap)

    recommendations = []
    for i in range(r):
        course = heapq.heappop(heap)
        recommendations.append((course[1], -course[0]))

    # return only top r courses in priority queue
    return recommendations


def getPredictedCourseRatings_opt(kNNs, users, user_id, courses, is_validation):
    """
    Get the r top recommended courses for a user
    :param: users: dictionary that maps user_id to User object, contains information of all users
    :param user_id: user id of the target user
    :param k: number of nearest neighbors
    :param threshold: commonality threshold
    :param comparatorSet: set of ids of target preferences
    :param r: number of recommendations to output
    :param rho: rho used to calculate smoothed prediction
    :param is_validation: whether the recommendation is for validation purposes or not
    :param pref_weight: weight of preference similarity between users to determine overall similarity
    :param course_weight: weight of course similarity between users to determine overall similarity
    :return: dictionary mapping course_id to predicted_rating for that course id
    """
    # a set of course_ids of courses rated by kNN
    rated_courses = set([])
    # a dictionary mapping course_id to a list of all ratings by kNN
    ratings = {}
    for neighbor_id in kNNs:
        # loop through all ratings by this neighbor
        for (course_id, rating) in users.get(neighbor_id).courseRatings.items():
            rated_courses.add(course_id)
            if course_id not in ratings.keys():
                ratings[course_id] = [rating]
            else:
                ratings[course_id].append(rating)

    # courses taken by the target user
    courses_taken = users.get(user_id).courseRatings.keys()

    if not is_validation:
        courses_to_rate = rated_courses.difference(courses_taken)
    else:
        courses_to_rate = rated_courses.intersection(courses_taken)

    # initialize priority queue, which will store (predicted_rating, course_id) pairs
    heap = []

    # for each course, predict the score and add it to the priority queue
    for course in courses_to_rate:
        n = len(ratings[course])
        average = 0
        for rating in ratings[course]:
            average += rating / n
        predicted_rating = (courses[course].average_rating + (n * average)) / (1 + n)
        heapq.heappush(heap, (-predicted_rating, course))

    recommendations = []
    while len(heap) > 0:
        course = heapq.heappop(heap)
        recommendations.append((course[1], -course[0]))

    # return only top r courses in priority queue
    return recommendations