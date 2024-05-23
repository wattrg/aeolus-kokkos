#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <gas/flow_state.h>
#include <grid/grid.h>
#include <util/cubic_spline.h>
#include <gas/transport_properties.h>

enum class BoundaryConditions { SupersonicInflow, SlipWall, SupersonicOutflow };

template <typename T>
class BoundaryAction {
public:
    virtual ~BoundaryAction() {}

    virtual void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
                       const Field<size_t>& boundary_faces, 
                       const IdealGas<T>& gas_model,
                       const TransportProperties<T>& trans_prop) = 0;
};

template <typename T>
class FlowStateCopy : public BoundaryAction<T> {
public:
    FlowStateCopy(FlowState<T> fs) : fs_(fs) {}

    FlowStateCopy(json flow_state);

    ~FlowStateCopy() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces, const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);

private:
    FlowState<T> fs_;
};

template <typename T>
class BoundaryLayerProfile : public BoundaryAction<T> {
public:
    BoundaryLayerProfile(json config);

    ~BoundaryLayerProfile() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces,
               const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);

private:
    CubicSpline v_;
    CubicSpline T_;
    T p_;
};

template <typename T>
class InternalCopy : public BoundaryAction<T> {
public:
    ~InternalCopy() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces,
               const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);
};

template <typename T>
class InternalCopyReflectNormal : public BoundaryAction<T> {
public:
    ~InternalCopyReflectNormal() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces,
               const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);
};

template <typename T>
class InternalVelCopyReflect : public BoundaryAction<T> {
public:
    ~InternalVelCopyReflect() {}

    void apply(FlowStates<T>& fs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces,
               const IdealGas<T>& gas_model,
               const TransportProperties<T>&trans_prop);
};

template <typename T>
class FixTemperature : public BoundaryAction<T> {
public:
    ~FixTemperature() {}

    FixTemperature(double temperature) : Twall_(temperature) {}

    void apply(FlowStates<T>& gs, const GridBlock<T>& grid,
               const Field<size_t>& boundary_faces,
               const IdealGas<T>& gas_model,
               const TransportProperties<T>& trans_prop);

private:
    double Twall_;
};

template <typename T>
class BoundaryCondition {
public:
    BoundaryCondition(json config);

    void apply_pre_reconstruction(FlowStates<T>& fs, const GridBlock<T>& grid,
                                  const Field<size_t>& boundary_faces,
                                  const IdealGas<T>& gas_model,
                                  const TransportProperties<T>& trans_prop);

    void apply_pre_viscous_grad(FlowStates<T>& fs, const GridBlock<T>& grid,
                                const Field<size_t>& boundary_faces,
                                const IdealGas<T>& gas_model,
                                const TransportProperties<T>& trans_prop);

private:
    std::vector<std::shared_ptr<BoundaryAction<T>>> pre_reconstruction_;
    std::vector<std::shared_ptr<BoundaryAction<T>>> pre_viscous_grad_;
};

#endif
