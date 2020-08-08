from ds.sum_tree import SumTree


def test_sum_tree():
    data = list(range(1, 100))
    capacity = 2 * len(data)

    power_of_two = 1
    while power_of_two <= capacity:
        power_of_two *= 2

    st = SumTree(power_of_two)

    assert st.capacity == power_of_two

    # set the initial data
    for i, x in enumerate(data):
        st[i] = x

    # test total sum
    assert st.total == sum(data), "wrong total sum"

    # test getting: random element and boundaries
    assert st[0] == data[0]
    assert st[4] == data[4]

    end = len(data) - 1
    assert st[end] == data[end]

    # test queries, including explicit [0, x], implicit [0, x], [x, x], [x, x+1], [end, end]
    def offline_query(dt, fr=0, to=None):
        if to is None:
            return sum(dt[fr:])

        return sum(dt[fr : to + 1])

    assert st.query() == st.total
    assert st.query() == sum(data)

    assert st.query(1) == offline_query(data, 1)
    assert st.query(15) == offline_query(data, 15)

    assert st.query(3, 18) == offline_query(data, 3, 18)
    assert st.query(9, 12) == offline_query(data, 9, 12)

    assert st.query(0, 0) == data[0]
    assert st.query(0, 1) == data[0] + data[1]

    assert st.query(end, end) == data[end]

    # test updates: random element and boundaries
    previous_sum = st.query(3, 18)
    change = 100
    st[4] = st[4] + change
    assert st.query(3, 18) == previous_sum + change

    previous_total = st.total
    st[end] = st[end] - 100
    st[0] = st[0] - 100
    assert st.total == previous_total - 200
