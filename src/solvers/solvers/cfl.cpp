#include "cfl.h"

#include <doctest/doctest.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <stdexcept>

ConstantSchedule::ConstantSchedule(Ibis::real cfl) : cfl_(cfl) {}

Ibis::real ConstantSchedule::eval(Ibis::real t) {
    (void)t;
    return cfl_;
}

LinearSchedule::LinearSchedule(json schedule) {
    std::vector<Ibis::real> times = schedule.at("times");
    std::vector<Ibis::real> cfls = schedule.at("cfls");
    times_ = times;
    cfls_ = cfls;
}

Ibis::real LinearSchedule::eval(Ibis::real t) {
    if (t <= times_[0]) {
        return cfls_[0];
    }

    size_t n = times_.size();
    if (t >= times_[n - 1]) {
        return cfls_[n - 1];
    }

    Ibis::real cfl;
    for (size_t i = 0; i < times_.size(); i++) {
        if (t >= times_[i] && t < times_[i + 1]) {
            Ibis::real frac = (t - times_[i]) / (times_[i + 1] - times_[i]);
            cfl = cfls_[i] + frac * (cfls_[i + 1] - cfls_[i]);
            return cfl;
        }
    }
    throw std::runtime_error("Shouldn't reach here");
}

ResidualBasedCfl::ResidualBasedCfl(Ibis::real threshold, Ibis::real power,
                                   Ibis::real start_cfl, Ibis::real max_cfl)
    : threshold_(threshold), power_(power), start_cfl_(start_cfl), max_cfl_(max_cfl),
      previous_cfl_(start_cfl), previous_residual_(1.0) {}

ResidualBasedCfl::ResidualBasedCfl(json config) 
    : ResidualBasedCfl(config.at("growth_threshold"), config.at("power"),
                       config.at("start_cfl"), config.at("max_cfl")) {}

Ibis::real ResidualBasedCfl::eval(Ibis::real t) {
    if (t > threshold_) { 
        previous_residual_ = t;
        return previous_cfl_;
    }
    // Ibis::real ratio = t / previous_residual_;
    Ibis::real ratio = previous_residual_ / t;
    Ibis::real new_cfl = previous_cfl_ * Ibis::pow(ratio, power_);
    new_cfl = Ibis::min(new_cfl, max_cfl_);
    new_cfl = Ibis::max(start_cfl_, new_cfl);
    previous_residual_ = t;
    previous_cfl_ = new_cfl;
    return new_cfl;
}


std::unique_ptr<CflSchedule> make_cfl_schedule(json config) {
    std::string type = config.at("type");
    if (type == "constant") {
        return std::unique_ptr<CflSchedule>(new ConstantSchedule(config.at("value")));
    } else if (type == "linear_interpolate") {
        return std::unique_ptr<CflSchedule>(new LinearSchedule(config));
    } else if (type == "residual_based") {
        return std::unique_ptr<CflSchedule>(new ResidualBasedCfl(config));
    } else {
        spdlog::error("Unkown CFL schedule {}", type);
        throw std::runtime_error("Unknown CFL schedule");
    }
}

TEST_CASE("CflSchedule") {
    json times = {0.0, 1.0, 2.0};
    json cfls = {0.1, 0.2, 0.5};
    json schedule_json = {{"times", times}, {"cfls", cfls}};

    LinearSchedule schedule{schedule_json};
    CHECK(schedule.eval(0.0) == doctest::Approx(0.1));
    CHECK(schedule.eval(0.5) == doctest::Approx(0.15));
    CHECK(schedule.eval(1.5) == doctest::Approx(0.35));
    CHECK(schedule.eval(2.5) == doctest::Approx(0.5));
}
