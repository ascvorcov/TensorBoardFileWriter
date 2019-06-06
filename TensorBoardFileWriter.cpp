#include <CNTKLibrary.h>

typedef struct {
  CNTK::Function* ptr;
} FunctionRef;

class TensorBoardProgressWriter : public CNTK::ProgressWriter
{
  CNTK::Internal::TensorBoardFileWriter writer;
  public:
  TensorBoardProgressWriter(size_t frequency, const std::wstring& dir, CNTK::Function* ptr)
    : CNTK::ProgressWriter(frequency, 1, 1, 1, 1, 1)
    , writer(dir, CNTK::FunctionPtr(ptr))
  {
  }

  double _avg(const std::pair<double, double>& numerator, const std::pair<size_t, size_t>& denominator)
  {
      auto num = numerator.second - numerator.first;
      auto den = denominator.second - denominator.first;
      return den > 0 ? (num / den) : 0.0;
  }

  double _avg(double num, size_t den) { return den > 0 ? (num / den) : 0.0; }

  void WriteValue(const std::wstring& name, float value, size_t step)
  {
      writer.WriteValue(name, value, (uint64_t)step);
  }

  virtual void OnWriteTrainingUpdate(const std::pair<size_t, size_t>& samples,
                                     const std::pair<size_t, size_t>& updates,
                                     const std::pair<double, double>& aggregateLoss,
                                     const std::pair<double, double>& aggregateMetric)
  {
      WriteValue(L"minibatch/avg_loss", _avg(aggregateLoss, samples), TotalTrainingUpdates());
      WriteValue(L"minibatch/avg_metric", _avg(aggregateMetric, samples), TotalTrainingUpdates());
  };

  virtual void OnWriteTestUpdate(const std::pair<size_t, size_t>& /*samples*/,
                                 const std::pair<size_t, size_t>& /*updates*/,
                                 const std::pair<double, double>& /*aggregateMetric*/) 
  {
      printf("TensorBoardProgressWriter does not support recording per-minibatch cross-validation results");
  };

  virtual void OnWriteTrainingSummary(size_t samples, size_t updates, size_t summaries,
                                      double aggregateLoss, double aggregateMetric,
                                      size_t elapsedMilliseconds)
  {
      WriteValue(L"summary/avg_loss", _avg(aggregateLoss, samples), summaries);
      WriteValue(L"summary/avg_metric", _avg(aggregateMetric, samples), summaries);
  };

  virtual void OnWriteTestSummary(size_t samples, size_t updates, size_t summaries,
                                        double aggregateMetric, size_t elapsedMilliseconds)
  {
      auto avg_metric = _avg(aggregateMetric, samples);
      if (TotalTrainingUpdates() != 0)
      {
          // Record test summary using training minibatches as a step.
          // This allows to easier correlate the training and test metric graphs in TensorBoard.
          WriteValue(L"minibatch/test_avg_metric", avg_metric, TotalTrainingUpdates());
      }
      else
      {
          WriteValue(L"summary/test_avg_metric", avg_metric, summaries);
      }
  };
};

#define API __declspec(dllexport)
extern "C" 
{
  API void InitVec(wchar_t* dir, FunctionRef* pf, void* progressVector)
  {
    CNTK::Function* ptr = (CNTK::Function*)pf->ptr;
    auto vec = (std::vector<CNTK::ProgressWriterPtr>*)progressVector;
    std::shared_ptr<CNTK::ProgressWriter> sp(new TensorBoardProgressWriter(1, dir, ptr));
    vec->push_back(sp);
  }
}
