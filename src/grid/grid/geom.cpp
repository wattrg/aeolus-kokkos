#include <doctest/doctest.h>
#include <grid/geom.h>

TEST_CASE("distance_between_points") {
    // allocate positions
    Vector3s<Ibis::real> pos{"pos", 10};
    auto pos_host = pos.host_mirror();
    for (int i = 0; i < 10; i++) {
        pos_host.x(i) = 3 * i;
        pos_host.y(i) = i + 1;
        pos_host.z(i) = 3 * i - 5;
    }
    pos.deep_copy(pos_host);

    // allocate the indicies we're going to query
    Field<int> i("i", 5);
    auto i_host = i.host_mirror();
    Field<int> j("j", 5);
    auto j_host = j.host_mirror();
    i_host(0) = 0;
    i_host(1) = 1;
    i_host(2) = 5;
    i_host(3) = 9;
    i_host(4) = 2;
    j_host(0) = 1;
    j_host(1) = 2;
    j_host(2) = 5;
    j_host(3) = 9;
    j_host(4) = 4;
    i.deep_copy(i_host);
    j.deep_copy(j_host);

    // do the calculations
    Field<Ibis::real> results("results", 5);
    Kokkos::parallel_for(
        "distance", 5, KOKKOS_LAMBDA(const int id) {
            results(id) = Ibis::distance_between_points(pos, i(id), j(id));
        });

    // copy results back to the host
    auto results_host = results.host_mirror();
    results_host.deep_copy(results);

    // check results
    CHECK(results_host(0) == doctest::Approx(Ibis::sqrt(19.0)));
}
