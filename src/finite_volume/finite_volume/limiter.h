#ifndef LIMITER_H
#define LIMITER_H

#include <gas/flow_state.h>
#include <grid/grid.h>
#include <util/types.h>
#include <util/vector3.h>

#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T>
struct LimiterValues {
    LimiterValues() {}

    LimiterValues(int num_cells, bool need_pressure, bool need_rho, bool need_temp,
                  bool need_u) {
        if (need_pressure) {
            p = Field<T>("Limiter::p", num_cells);
        }
        if (need_temp) {
            temp = Field<T>("Limiter::T", num_cells);
        }
        if (need_rho) {
            rho = Field<T>("Limiter::rho", num_cells);
        }
        if (need_u) {
            u = Field<T>("Limtier::u", num_cells);
        }
        vx = Field<T>("Limiter::vx", num_cells);
        vy = Field<T>("Limiter::vy", num_cells);
        vz = Field<T>("Limiter::vz", num_cells);
    }

    Field<T> p;
    Field<T> rho;
    Field<T> temp;
    Field<T> u;
    Field<T> vx;
    Field<T> vy;
    Field<T> vz;
};

template <typename T>
class Limiter {
public:
    virtual ~Limiter() {}

    Limiter() : enabled_(true) {}

    Limiter(bool enabled) : enabled_(enabled) {}

    virtual void calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits,
                                    const Cells<T>& cells, const Interfaces<T>& faces,
                                    Vector3s<T>& grad) = 0;

    KOKKOS_INLINE_FUNCTION
    bool enabled() const { return enabled_; }

private:
    bool enabled_;
};

template <typename T>
class Unlimited : public Limiter<T> {
public:
    Unlimited() : Limiter<T>(false) {}

    ~Unlimited() {}

    void calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits,
                            const Cells<T>& cells, const Interfaces<T>& faces,
                            Vector3s<T>& grad);
};

template <typename T>
class BarthJespersen : public Limiter<T> {
public:
    ~BarthJespersen() {}

    BarthJespersen(Ibis::real epsilon) : Limiter<T>(true), epsilon_(epsilon) {}

    void calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits,
                            const Cells<T>& cells, const Interfaces<T>& faces,
                            Vector3s<T>& grid);

private:
    Ibis::real epsilon_;
};

template <typename T>
std::unique_ptr<Limiter<T>> make_limiter(json config);

#endif
