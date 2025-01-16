#include "HistoToCTRegistrationAlgorithm.h"

#include <ImFusion/Base/BasicImageProcessing.h>
#include <ImFusion/Base/CombineImagesAsVolumeAlgorithm.h>
#include <ImFusion/Base/ImageResamplingAlgorithm.h>
#include <ImFusion/Base/OptimizerNL.h>
#include <ImFusion/Base/Pose.h>
#include <ImFusion/Base/Progress.h>
#include <ImFusion/Base/SharedImageSet.h>
#include <ImFusion/Base/Utils/AlgorithmUtils.h>
#include <ImFusion/Core/Log.h>
#include <ImFusion/Core/Random.h>
#include <ImFusion/Ext/fmt/format.h>
#include <ImFusion/GL/BakeDeformationAlgorithm.h>
#include <ImFusion/GL/BakeTransformationAlgorithm.h>
#include <ImFusion/GL/Deformation.h>
#include <ImFusion/GL/FreeFormDeformation.h>
#include <ImFusion/Histology/HistologyMicroCTPreprocessingAlgorithm.h>
#include <ImFusion/Histology/TiledImageSet.h>
#include <ImFusion/ImageMath/SharedImageSetArithmetic.h>
#include <ImFusion/ML/DISAFeaturesAlgorithm.h>
#include <ImFusion/ML/Operations.h>
#include <ImFusion/Reg/FFDImageRegistration.h>
#include <ImFusion/Reg/FeatureMapsRegistrationAlgorithm.h>
#include <ImFusion/Reg/ImageRegistration.h>
#include <ImFusion/Reg/SimilarityMeasureWrapper.h>

#include <filesystem>

namespace ImFusion
{
	namespace IM = ImageMath;

	namespace
	{
		std::shared_ptr<Optimizer> createOptimizer(double stepSize = 0.1)
		{
			LOG_INFO("Initialize default optimizer");
			auto opt = std::make_shared<OptimizerNL>(3, 34, nullptr);
			opt.get()->setStep(stepSize);
			opt.get()->setLogging(0, 2);        // logging every evaluation
			opt.get()->setAbortParTol(1e-4);    // set parameter tolerance
			opt.get()->setBounds(std::vector<double>({1, 1, 1}));
			opt.get()->setSelection(std::vector<bool>({1, 1, 0}));    // optimize over translation parameters
			opt.get()->setMinimize(false);                            // optimization direction minimize/maximize
			opt.get()->setProgressUpdate(true);
			opt.get()->setProgressUpdatesDisplay(true);
			return opt;
		}

		class HistologyMicroCTSliceCostFunction : public CostFunction
		{
		public:
			HistologyMicroCTSliceCostFunction(SharedImageSet& histology,
											  SharedImageSet& volume,
											  const vec3& planePoint,
											  const mat4 initialMat,
											  FreeFormDeformation* def = nullptr,
											  const mat4 initialPlaneMat = mat4::Zero(),
											  bool is3D = false);

			std::array<double, 6> currentParams();
			mat4 bestMat();
			double evaluate(int n, const double* x, double* dx = 0);


		private:
			SharedImageSet& m_histology;
			SharedImageSet& m_volume;
			std::shared_ptr<FreeFormDeformation> m_def;
			mat4 m_matrix;
			std::unique_ptr<ImageDescriptorWorld> m_sliceDesc;
			SimilarityMeasureWrapper m_sm = {nullptr, nullptr, 2};
			bool m_is3D;
		};

		HistologyMicroCTSliceCostFunction::HistologyMicroCTSliceCostFunction(SharedImageSet& histology,
																			 SharedImageSet& volume,
																			 const vec3& planePoint,
																			 const mat4 initialMat,
																			 FreeFormDeformation* def,
																			 const mat4 initialPlaneMat,
																			 bool is3D)
			: m_histology(histology)
			, m_volume(volume)
			, m_matrix(initialMat)
			, m_def(def)
			, m_is3D(is3D)
		{
			int maxDim = m_volume.get(0)->dimensions().maxCoeff();
			double minSpacing = m_volume.get(0)->spacing().minCoeff();
			ImageDescriptor sliceDesc(PixelType::Float, vec3i(maxDim, maxDim, 1));
			sliceDesc.setSpacing(vec3(minSpacing, minSpacing, minSpacing), true);

			// clang-format off
			mat4 sliceMat = (mat4() <<
									1,  0,  0,  0,
									0,  0,  1,  0,
									0, -1,  0,  0, 
									0,  0,  0,  1).finished(); // slice along y
			// clang-format on
			sliceMat.topRightCorner<3, 1>() = vec3({0.0, m_volume.get()->pixelToWorld(planePoint)[1], 0.0});
			if (initialPlaneMat != mat4::Zero())
				sliceMat = initialPlaneMat;
			m_sliceDesc = std::make_unique<ImageDescriptorWorld>(sliceDesc, sliceMat);

			// Downsampling before registration
			ImageResamplingAlgorithm resampling(nullptr);
			resampling.p_resamplingMode = ImageResamplingAlgorithm::TARGET_PERCENT;
			int percent = 50;
			resampling.p_targetPercent = vec3i(percent, percent, percent);
			resampling.setSequentialMode(true);
			m_histology.replace(0, resampling.processImage(*m_histology.get(), m_histology.components()).first);
		}

		std::array<double, 6> HistologyMicroCTSliceCostFunction::currentParams()
		{
			std::array<double, 6> params;
			Pose::matToEuler(m_sliceDesc->matrixToWorld(), params.data());

			// Factor to the angles to cope with mm vs degree
			params[3] /= 10.0;
			params[4] /= 10.0;
			params[5] /= 10.0;

			return params;
		}

		mat4 HistologyMicroCTSliceCostFunction::bestMat() { return m_matrix; }

		double HistologyMicroCTSliceCostFunction::evaluate(int n, const double* x, double* dx)
		{
			// 2D deformation (in-plane deformation only)
			if (!m_is3D)
			{
				vec3 t = vec3::Zero(), r = vec3::Zero();
				t << x[0], x[1], x[2];
				r << x[3], x[4], x[5];

				// Factor to the angles to cope with mm vs degree
				r *= 10.0;
				m_sliceDesc->setMatrixToWorld(Pose::eulerToMat(t, r));

				// Use ImageMath resample along y-axis
				auto slice = IM::makeArray(m_volume /*, false, IM::MagFilter::Linear, IM::Wrap::ClampToBorder, vec4f::Zero()*/)
								 .resample(m_volume.get()->descriptorWorld(), *m_sliceDesc)
								 .evaluateIntoImage<float>(true);

				slice = ML::NormalizePercentileOperation(0.02, 0.98, true).process(std::move(slice));

				auto histoTemp = m_histology.clone();
				auto reg = std::make_unique<ImageRegistration>(slice.get(), histoTemp.get());
				reg->parametric()->setSimilarityMeasure(SimilarityMeasureFactory::Mode::NCC);
				reg->setTransformationModel(ImageRegistration::TransformationModel::Linear);
				reg->compute();
				reg->parametric()->setAffine(true);
				reg->optimizer()->setSelection(std::vector<bool>({1, 1, 1, 1, 1, 1}));    // scale x/y
				reg->parametric()->setSimilarityMeasure(SimilarityMeasureFactory::Mode::LNCC);
				reg->compute();

				m_matrix = histoTemp->matrix().inverse();

				reg->parametric()->setAffine(false);
				reg->setTransformationModel(ImageRegistration::TransformationModel::FFD);

				auto sliceDef = std::dynamic_pointer_cast<FreeFormDeformation>(histoTemp->deformation());
				sliceDef->setGridTransformation(m_matrix.inverse());
				vec3i subdivisions = vec3i(5, 5, 1);
				sliceDef->setSubdivisions(subdivisions);

				reg->optimizer()->setDimension(72);    // 16 control points * (x,y) coordinates
				reg->optimizer()->setSelection(std::vector<bool>(reg->optimizer()->dimension(), 1));
				reg->compute();

				m_def = sliceDef;

				double result = reg->bestSimilarityValue();
				return result;
			}
			// 3D deformation (Apply 3D deformation on the volume, then extract slice for registration)
			else
			{
				std::shared_ptr<FreeFormDeformation> volDef = std::dynamic_pointer_cast<FreeFormDeformation>(m_volume.deformation());
				for (int xx = 0; xx < 4; xx++)
					for (int zz = 0; zz < 4; zz++)
						for (int yy = 0; yy < 2; yy++)
							volDef->setDisplacement(vec3i(xx, yy, zz), vec3f(0, static_cast<float>(x[xx * 4 + zz]), 0));

				auto bakeDef = BakeDeformationAlgorithm(m_volume);
				bakeDef.compute();
				auto newVol = bakeDef.takeOutput().extractFirstImage();
				// Use ImageMath resample along y-axis to extract CT slices
				auto slice = IM::makeArray(*newVol /*, false, IM::MagFilter::Linear, IM::Wrap::ClampToBorder, vec4f::Zero()*/)
								 .resample(newVol->get()->descriptorWorld(), *m_sliceDesc)
								 .evaluateIntoImage<float>(true);

				slice = ML::NormalizePercentileOperation(0.02, 0.98, true).process(std::move(slice));
				m_histology.get()->setMatrixToWorld(m_matrix);
				auto histoTemp = m_histology.clone();
				auto reg = std::make_unique<ImageRegistration>(slice.get(), histoTemp.get());
				reg->setOptimizer(createOptimizer(1.0));
				reg->parametric()->setSimilarityMeasure(SimilarityMeasureFactory::Mode::LNCC);
				reg->setTransformationModel(ImageRegistration::TransformationModel::Linear);
				reg->parametric()->setAffine(true);    // set to affine registration for soft tissue block registration
				reg->optimizer()->setSelection(std::vector<bool>({1, 1, 1, 1, 1, 1}));    // scale x/y
				reg->compute();
				reg->parametric()->setAffine(false);
				reg->setTransformationModel(ImageRegistration::TransformationModel::FFD);

				auto sliceDef = std::dynamic_pointer_cast<FreeFormDeformation>(histoTemp->deformation());
				sliceDef->setGridTransformation(m_matrix.inverse());
				vec3i subdivisions = vec3i(5, 5, 1);
				sliceDef->setSubdivisions(subdivisions);

				reg->optimizer()->setDimension(72);    // 16 control points * (x,y) coordinates
				reg->optimizer()->setSelection(std::vector<bool>(reg->optimizer()->dimension(), 1));
				reg->compute();

				m_def = sliceDef;

				double result = reg->bestSimilarityValue();

				return result;
			}
		}
	}

	namespace Histology
	{
		bool HistoToCTRegistrationAlgorithm::createCompatible(const DataList& dl, Algorithm** a)
		{
			if (dl.size() != 2)
				return false;
			TiledImageSet* tis = dl.getFirst<TiledImageSet>();
			if (tis)
			{
				auto sis = dynamic_cast<SharedImageSet*>(tis == dl[0] ? dl[1] : dl[0]);
				if (!sis)
					return false;
				if (a)
					*a = new HistoToCTRegistrationAlgorithm(*tis, *sis);
				return true;
			}
			else
			{
				auto vol = dl.getImage(Data::VOLUME);
				if (!vol)
					return false;

				SharedImageSet* histo = dl.getImage(Data::IMAGE);
				if (a)
					*a = new HistoToCTRegistrationAlgorithm(*histo, *vol);
				return true;
			}
		}

		HistoToCTRegistrationAlgorithm::HistoToCTRegistrationAlgorithm(TiledImageSet& tis, SharedImageSet& vol)
			: m_tis(&tis)
			, m_vol(&vol)
		{
			p_level.setRange(0, m_tis->numLevels() - 1);
			p_level = m_tis->numLevels() - 3;
			p_initMode.setType(Properties::ParamType::Enum);
			p_initMode.setAttribute("values", "DISA, Manual");
			p_preprocessingAlgorithm = std::make_unique<HistologyMicroCTPreprocessingAlgorithm>(*m_tis, *m_vol);

			std::function<void(void)> configParams = [this]() {
				if (p_initMode.value() == InitMode::DISA)
				{
					p_disaTranslationRange.setAttribute("hidden", std::to_string(false));
					p_disaRotationRange.setAttribute("hidden", std::to_string(false));
				}
				else
				{
					p_disaTranslationRange.setAttribute("hidden", std::to_string(true));
					p_disaRotationRange.setAttribute("hidden", std::to_string(true));
				}
			};
			configParams();
			p_initMode.signalValueChanged.connect(configParams);
		}

		HistoToCTRegistrationAlgorithm::HistoToCTRegistrationAlgorithm(SharedImageSet& histo, SharedImageSet& vol)
			: m_histo(&histo)
			, m_vol(&vol)
		{
			p_initMode.setType(Properties::ParamType::Enum);
			p_initMode.setAttribute("values", "DISA, Manual");
			p_preprocessingAlgorithm = std::make_unique<HistologyMicroCTPreprocessingAlgorithm>(*m_histo, *m_vol);

			std::function<void(void)> configParams = [this]() {
				if (p_initMode.value() == InitMode::DISA)
				{
					p_disaTranslationRange.setAttribute("hidden", std::to_string(false));
					p_disaRotationRange.setAttribute("hidden", std::to_string(false));
				}
				else
				{
					p_disaTranslationRange.setAttribute("hidden", std::to_string(true));
					p_disaRotationRange.setAttribute("hidden", std::to_string(true));
				}
			};
			configParams();
			p_initMode.signalValueChanged.connect(configParams);
		}

		HistoToCTRegistrationAlgorithm::~HistoToCTRegistrationAlgorithm() {}

		void HistoToCTRegistrationAlgorithm::compute()
		{
			m_status = Status::Error;

			planeInitialization();
			compute2D3DRegistration();

			m_status = Status::Success;
		}

		void HistoToCTRegistrationAlgorithm::planeInitialization()
		{
			m_status = Status::Error;
			m_sliceMat = mat4::Identity();

			/****************** Preprocess images for feature extraction ******************/
			if (m_tis)
			{
				if (p_level.value() < 0 || p_level.value() >= m_tis->numLevels())
					return;
				const ImagePyramidLevel& pyramidLevel = m_tis->level(p_level);
				auto histoSlideLevel = std::make_unique<SharedImageSet>(pyramidLevel.subImage(vec2i(0, 0), m_tis->level(p_level).dimensions()));
				m_preprocessedHisto = IM::makeArray(*histoSlideLevel).evaluateIntoImage<float>(true);
			}
			else
			{
				m_preprocessedHisto = IM::makeArray(*m_histo).evaluateIntoImage<float>(true);
			}

			m_preprocessedHisto = p_preprocessingAlgorithm.value()->preprocessedHisto(m_preprocessedHisto.get());

			auto volDesc = m_vol->get()->descriptor();
			// clang-format off
			mat4 resampleMat = (mat4() << 
							   1,  0,  0,  0, 
						       0,  1,  0,  0, 
							   0,  0,  1,  0, 
							   0,  0,  0,  1).finished();
			// clang-format on
			m_vol->setMatrixFromWorld(resampleMat);
			m_vol->get()->setSpacing(vec3(1.0, 1.0, 1.0));
			m_preprocessedVol = m_vol->clone();
			m_preprocessedVol = p_preprocessingAlgorithm.value()->preprocessedCt(m_preprocessedVol.get());

			ImageDescriptor sliceDesc(PixelType::Float, vec3i(volDesc.width, volDesc.slices, 1));
			sliceDesc.setSpacing(vec3(1.0, 1.0, 1.0), true);

			if (p_initMode == InitMode::DISA)
			{
				computeFeatureMap(*m_preprocessedHisto, *m_preprocessedVol);
			}

			// clang-format off
			mat4 resampleSliceMat = (mat4() <<
									1,  0,  0,  0,
									0,  0,  1,  0,
									0, -1,  0,  0, 
									0,  0,  0,  1).finished(); // slice along y
			// clang-format on
			ImageDescriptorWorld sliceDescWorld(sliceDesc, resampleSliceMat);

			int sliceUpper = 130;
			int currentSlice = 0;
			double currentMax = 0.0;
			std::shared_ptr<FreeFormDeformation> def;

			// Compute the rough transformation matrix (2D) of histo slide and prepare for the patches
			std::array<double, 6> avgEuler;
			for (int slice = p_sliceBound.value()[0]; slice < p_sliceBound.value()[1]; slice++)
			{
				resampleSliceMat.block<3, 1>(0, 3) = vec3({0.0, m_preprocessedVol->get()->pixelToWorld(vec3(0, slice * 1.0, 0))[1], 0.0});
				sliceDescWorld.setMatrixToWorld(resampleSliceMat);
				auto volSlice = IM::makeArray(*m_preprocessedVol)
									.resample(m_preprocessedVol->get(0)->descriptorWorld(), sliceDescWorld)
									.evaluateIntoImage<float>(true);

				volSlice->get()->setSpacing(vec3(1.0, 1.0, 1.0));    // set a dummy spacing to cope with the numerical limit
				m_imgReg = std::make_unique<ImageRegistration>(volSlice.get(), m_preprocessedHisto.get());

				m_imgReg->setMoving(volSlice.get());          // move CT slice
				m_imgReg->setOptimizer(createOptimizer());    // set to OptimizerNL
				dynamic_cast<OptimizerNL*>(m_imgReg->optimizer().get())->setStep(15.0);
				m_imgReg->optimizer()->setBounds(std::vector<double>({100, 100, 200}));    // bound for soft tissue block registration
				m_imgReg->optimizer()->setSelection(std::vector<bool>({0, 0, 1}));         // rotation

				m_imgReg->parametric()->setSimilarityMeasure(SimilarityMeasureFactory::Mode::NCC);    // set similarity measure
				m_imgReg->setTransformationModel(ImageRegistration::TransformationModel::Linear);
				m_imgReg->compute();    // perform registration

				dynamic_cast<OptimizerNL*>(m_imgReg->optimizer().get())->setStep(10.0);
				m_imgReg->optimizer()->setSelection(std::vector<bool>({1, 1, 0, 0, 0, 0}));    // translation
				m_imgReg->compute();

				m_imgReg->optimizer()->setSelection(std::vector<bool>({0, 0, 1, 0, 0, 0}));    // rotation
				m_imgReg->compute();

				m_imgReg->optimizer()->setSelection(std::vector<bool>({1, 1, 0, 0, 0, 0}));    // translation
				m_imgReg->compute();                                                           // perform registration

				m_imgReg->parametric()->setAffine(true);    // set to affine registration for soft tissue block registration
				dynamic_cast<OptimizerNL*>(m_imgReg->optimizer().get())->setStep(1.0);
				m_imgReg->optimizer()->setSelection(std::vector<bool>({0, 0, 0, 1, 1, 0}));    // scale x/y
				m_imgReg->compute();                                                           // perform registration

				m_imgReg->parametric()->setAffine(false);

				m_imgReg->setTransformationModel(ImageRegistration::TransformationModel::FFD);
				dynamic_cast<FFDImageRegistration*>(m_imgReg->parametric())->setSmoothness(1e-3);
				m_imgReg->optimizer()->setDimension(32);    // 16 control points * (x,y) coordinates
				m_imgReg->optimizer()->setSelection(std::vector<bool>(m_imgReg->optimizer()->dimension(), 1));
				m_imgReg->compute();

				std::array<double, 6> params;
				Pose::matToEuler(volSlice->matrixToWorld(), params.data());

				if (slice == p_sliceBound.value()[0])
				{
					currentMax = m_imgReg->optimizer()->bestVal();
					avgEuler = params;
				}
				else
				{
					for (int i = 0; i < 6; i++)
						avgEuler[i] = (avgEuler[i] + params[i]) / 2.0;    // Take the average of the translation and rotation;
					if (currentMax < m_imgReg->optimizer()->bestVal() && slice > p_sliceBound.value()[0] &&
						slice < p_sliceBound.value()[1])    // save SM value of potential matched CT slices, ignore some top/bottom slices
					{
						currentMax = m_imgReg->optimizer()->bestVal();
						currentSlice = slice;
						m_2dRegMat = volSlice->matrixToWorld();
						def = dynamic_cast<FFDImageRegistration*>(m_imgReg->parametric())->deformation();
					}
				}
			}

			if (p_initMode == InitMode::MANUAL)
				return;

			// Apply the initial transformation to histo slide and bake transformation before extracting patches.
			mat4 initMat = Pose::eulerToMat(avgEuler.data());
			m_preprocessedHisto->setMatrixToWorld(
				initMat.inverse());    // Need to inverse the matrix since it is actually the transformation matrix of the CT slice.
			auto bakeTrans = BakeTransformationAlgorithm(m_preprocessedHisto.get());

			bakeTrans.compute();
			auto bakedHistoSlide = bakeTrans.takeOutput().extractFirstImage();

			sliceDesc = ImageDescriptor(PixelType::Float, vec3i(300, 300, 1));
			sliceDesc.setSpacing(vec3(1.0, 1.0, 1.0), true);
			ImageDescriptor slideDesc(PixelType::Float, vec3i(300, 300, 1));
			// clang-format off
			resampleSliceMat = (mat4() <<
									1,  0,  0,  0,
									0,  0,  1,  0,
									0, -1,  0,  0, 
									0,  0,  0,  1).finished(); // slice along y
			// clang-format on

			computeDISAInit();
		}

		void HistoToCTRegistrationAlgorithm::compute2D3DRegistration()
		{
			m_status = Status::Error;
			if (m_tis)
			{
				if (p_level.value() < 0 || p_level.value() >= m_tis->numLevels())
					return;
				//p_level = 2;
				const ImagePyramidLevel& pyramidLevel = m_tis->level(p_level);
				auto histoSlideLevel = std::make_unique<SharedImageSet>(pyramidLevel.subImage(vec2i(0, 0), m_tis->level(p_level).dimensions()));
				m_preprocessedHisto = IM::makeArray(*histoSlideLevel).evaluateIntoImage<float>(true);
			}
			else
			{
				m_preprocessedHisto = IM::makeArray(*m_histo).evaluateIntoImage<float>(true);
			}

			p_preprocessingAlgorithm.value()->p_convertToGray = true;
			m_preprocessedHisto = p_preprocessingAlgorithm.value()->preprocessedHisto(m_preprocessedHisto.get());

			auto volDesc = m_vol->get()->descriptor();
			// clang-format off
			mat4 resampleMat = (mat4() << 
									1, 0, 0, 0, 
									0, 1, 0, 0, 
									0, 0, 1, 0, 
									0, 0, 0, 1).finished();
			// clang-format on
			m_vol->setMatrixFromWorld(resampleMat);
			m_vol->get()->setSpacing(vec3(1.0, 1.0, 1.0));
			m_preprocessedVol = m_vol->clone();
			m_preprocessedVol = p_preprocessingAlgorithm.value()->preprocessedCt(m_preprocessedVol.get());
			ImageDescriptor sliceDesc(PixelType::Float, vec3i(volDesc.width + 100, volDesc.slices + 100, 1));
			sliceDesc.setSpacing(vec3(1.0, 1.0, 1.0), true);
			// clang-format off
			mat4 resampleSliceMat = (mat4() <<
									1,  0,  0,  0,
									0,  0,  1,  0,
									0, -1,  0,  0, 
									0,  0,  0,  1).finished(); // slice along y
			// clang-format on
			ImageDescriptorWorld sliceDescWorld(sliceDesc, resampleSliceMat);

			int sliceUpper = 130;
			int currentSlice = 0;
			double currentMax = 0.0;
			std::shared_ptr<FreeFormDeformation> def;

			// optimize over plane params
			// Initialize the optimization with 10 random planes that are close to the initialization plane, choose the one with the highest SM
			Progress::Task pt1(m_progress, std::max(p_numRandomSearch.value(), 1), "2D-3D registration");
			std::array<double, 6> bestParams;
			double bestValTmp = 0.0;
			for (int i = 0; i < std::max(p_numRandomSearch.value(), 1); i++)
			{
				pt1.setCurrentStep(i);
				if (pt1.wasCanceled())
					break;

				Random::Generator gen;

				mat4 randomTrans = mat4::Identity();
				// Introduce random shift and rotation to the initial plane
				if (p_numRandomSearch.value() > 0)
				{
					vec3 t = vec3::Zero(), r = vec3::Zero();
					t[1] = gen.getUniformReal(-3.0, 3.0);
					r << gen.getUniformReal(-3.0, 3.0), 0.0, gen.getUniformReal(-3.0, 3.0);
					mat4 randomTrans = Pose::eulerToMat(t, r);
				}

				auto histoTemp = m_preprocessedHisto->clone();
				HistologyMicroCTSliceCostFunction* planeCostFunc;

				planeCostFunc = new HistologyMicroCTSliceCostFunction(
					*histoTemp.get(), *m_preprocessedVol, vec3::Zero(), m_2dRegMat, nullptr, m_sliceMat * randomTrans, false);


				auto opt = std::make_shared<OptimizerNL>(6, 34, nullptr);
				opt->setStep(5.0);

				opt->setLogging(0, 2);
				opt->setAbortParTol(1e-4);
				opt->setMinimize(false);
				opt->setProgressUpdate(true);
				opt->setProgressUpdatesDisplay(true);
				opt->setCostFunction(planeCostFunc);
				opt->setSelection(std::vector<bool>({0, 1, 0, 0, 0, 0}));
				std::array<double, 6> currParams = planeCostFunc->currentParams();
				opt->execute(&currParams[0]);
				opt->evaluate(6, &currParams[0]);
				opt->setSelection(std::vector<bool>({0, 1, 0, 1, 0, 1}));
				opt->setStep(1.0);
				currParams = planeCostFunc->currentParams();
				opt->execute(&currParams[0]);
				opt->evaluate(6, &currParams[0]);
				if (opt->bestVal() > bestValTmp)
				{
					bestParams = currParams;
					bestValTmp = opt->bestVal();
					LOG_INFO("Current best cost function value:" + std::to_string(bestValTmp));
				}

				m_2dRegMat = planeCostFunc->bestMat();
			}
			vec3 t = vec3::Zero(), r = vec3::Zero();
			t << bestParams[0], bestParams[1], bestParams[2];
			r << bestParams[3], bestParams[4], bestParams[5];
			r *= 10.0;
			mat4 currOptMat = Pose::eulerToMat(t, r);

			setSliceMat(currOptMat);

			sliceDescWorld.setMatrixToWorld(currOptMat);
			m_optimalSlice1 = IM::makeArray(*m_vol, false, IM::MagFilter::Linear, IM::Wrap::ClampToEdge, vec4f::Zero())
								  .resample(m_vol->get()->descriptorWorld(), sliceDescWorld)
								  .evaluateIntoImage<float>(true);

			// Optimization of deformation in 3D
			def = std::make_shared<FreeFormDeformation>(vec3i(3, 1, 3), m_vol->get());
			m_vol->setDeformation(def);
			m_preprocessedVol->setDeformation(def);
			std::array<double, 16> bestParams3d = {};
			bestValTmp = 0.0;

			// Initialize the optimization with 10 random planes that are close to the initialization plane, choose the one with the highest SM after 20 iterations
			Progress::Task pt2(m_progress, std::max(p_numRandomSearch.value(), 1), "3D deformable registration");
			for (int i = 0; i < std::max(p_numRandomSearch.value(), 1); i++)
			{
				pt2.setCurrentStep(i);
				if (pt2.wasCanceled())
					break;

				Random::Generator gen;

				std::array<double, 16> currentParams3d = {0};
				// Introduce random deformation to the initial plane
				if (p_numRandomSearch.value() > 0)
				{
					for (int j = 0; j < currentParams3d.size(); j++)
						currentParams3d[j] = gen.getUniformReal(-10.0, 10.0);
				}

				auto histoTemp = m_preprocessedHisto->clone();

				auto costFunc =
					new HistologyMicroCTSliceCostFunction(*histoTemp.get(), *m_preprocessedVol, vec3::Zero(), m_2dRegMat, nullptr, currOptMat, true);
				auto opt = std::make_shared<OptimizerNL>(16, 34, nullptr);
				opt->setStep(10.0);
				opt->setLogging(0, 2);
				opt->setAbortEval(20);
				opt->setMinimize(false);
				opt->setProgressUpdate(true);
				opt->setProgressUpdatesDisplay(true);
				opt->setCostFunction(costFunc);
				std::vector<bool> sel(16, 1);
				opt->setSelection(sel);
				opt->execute(&currentParams3d[0]);
				opt->evaluate(16, &currentParams3d[0]);

				if (opt->bestVal() > bestValTmp)
				{
					bestParams3d = currentParams3d;
					bestValTmp = opt->bestVal();
					LOG_INFO("Current best cost function value:" + std::to_string(bestValTmp));
				}
			}

			for (int xx = 0; xx < 4; xx++)
				for (int zz = 0; zz < 4; zz++)
					for (int yy = 0; yy < 2; yy++)
						def->setDisplacement(vec3i(xx, yy, zz), vec3f(0, static_cast<float>(bestParams3d[xx * 4 + zz]), 0));

			auto bakeDef = BakeDeformationAlgorithm(*m_vol);
			bakeDef.compute();
			auto newVol = bakeDef.takeOutput().extractFirstImage();

			m_optimalSlice2 = IM::makeArray(*newVol, false, IM::MagFilter::Linear, IM::Wrap::ClampToEdge)
								  .resample(newVol->get()->descriptorWorld(), sliceDescWorld)
								  .evaluateIntoImage<float>(true);


			m_status = Status::Success;
		}

		void HistoToCTRegistrationAlgorithm::computeFeatureMap(SharedImageSet& histoSlide, SharedImageSet& vol)
		{
			auto volDesc = vol.get()->descriptor();
			ImageDescriptor sliceDesc(PixelType::Float, vec3i(volDesc.width, volDesc.slices, 1));
			mat4 resampleSliceMat = (mat4() << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1).finished();    // slice along y
			// clang-format on
			ImageDescriptorWorld sliceDescWorld(sliceDesc, resampleSliceMat);
			auto sis = std::make_unique<SharedImageSet>();

			Progress::Task pt1(m_progress, volDesc.height, "DISA feature map generation...");
			for (int slice = 0; slice < volDesc.height; slice++)
			{
				pt1.setCurrentStep(slice);
				if (pt1.wasCanceled())
					break;

				else
				{
					resampleSliceMat.topRightCorner<3, 1>() = vec3({0.0, vol.get()->pixelToWorld(vec3(0, slice, 0))[1], 0.0});
				}
				sliceDescWorld.setMatrixToWorld(resampleSliceMat);
				auto volSlice = IM::makeArray(vol, false, IM::MagFilter::Linear, IM::Wrap::ClampToBorder, vec4f::Zero())
									.resample(vol.get(0)->descriptorWorld(), sliceDescWorld)
									.evaluateIntoImage<float>(true);
				sis->add(volSlice->getShared());
			}

			// Generate feature maps for CT
			{
				DISAFeaturesAlgorithm disaFeatureAlg(*sis);
				std::filesystem::path filePath(__FILE__);
				std::filesystem::path parentPath = filePath.parent_path().parent_path();
				disaFeatureAlg.setCustomModel(parentPath.string() + "\\HistoToCTRegistrationPlugin\\model\\model-Torch.yaml");
				disaFeatureAlg.compute();
				auto featureMaps = disaFeatureAlg.takeOutput().extractFirstImage();
				CombineImagesAsVolumeAlgorithm combineAlg(*featureMaps, volDesc.spacing[1]);
				combineAlg.p_axis = ImageProcessing::Axis::Y;
				combineAlg.compute();
				m_featureMapVol = combineAlg.takeOutput().extractFirstImage();
				m_featureMapVol->setMatrix(mat4::Identity());
			}
			// Generate feature maps for histo slide
			{
				DISAFeaturesAlgorithm disaFeatureAlg(histoSlide);
				std::filesystem::path filePath(__FILE__);
				std::filesystem::path parentPath = filePath.parent_path().parent_path();
				disaFeatureAlg.setCustomModel(parentPath.string() + "\\HistoToCTRegistrationPlugin\\model\\model-Torch.yaml");
				disaFeatureAlg.compute();
				m_featureMapHisto = disaFeatureAlg.takeOutput().extractFirstImage();
			}
		}

		void HistoToCTRegistrationAlgorithm::computeDISAInit()
		{
			if (!m_featureMapVol || !m_featureMapHisto)
				computeFeatureMap(*m_preprocessedHisto, *m_preprocessedVol);

			auto featureMapHistoVol = std::make_unique<SharedImageSet>();
			featureMapHistoVol->add(m_featureMapHisto->clone()->getShared());
			featureMapHistoVol->add(m_featureMapHisto->clone()->getShared());
			CombineImagesAsVolumeAlgorithm combineAlg(*featureMapHistoVol, m_preprocessedVol->get()->spacing()[1]);
			combineAlg.p_axis = ImageProcessing::Axis::Y;
			combineAlg.compute();
			featureMapHistoVol = combineAlg.takeOutput().extractFirstImage();

			auto volDesc = m_preprocessedVol->get()->descriptor();
			ImageDescriptor sliceDesc(PixelType::Float, vec3i(volDesc.width, volDesc.slices, 1));
			mat4 resampleSliceMat = (mat4() << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1).finished();    // slice along y
			// clang-format on
			ImageDescriptorWorld sliceDescWorld(sliceDesc, resampleSliceMat);

			mat4 optHistoMat;
			double currentMax = 0.0;

			vec2 rangeWorld = vec2(-m_vol->get()->pixelToWorld(vec3(0, p_sliceBound.value()[0] * 1.0, 0))[1],
								   -m_vol->get()->pixelToWorld(vec3(0, p_sliceBound.value()[1] * 1.0, 0))[1]);

			Progress::Task pt2(m_progress, 10, "Initial plane searching...");
			for (int i = 0; i < 10; i++)
			{
				pt2.setCurrentStep(i);
				if (pt2.wasCanceled())
					break;

				std::array<double, 6> params;
				Pose::matToEuler(m_2dRegMat, params.data());
				std::array<double, 2> paramsTemp({params[0], params[4]});    // Need to switch the y and z rotation from m_2dRegMat
				params[0] = params[1];
				params[1] = rangeWorld[1] - i * (rangeWorld[1] - rangeWorld[0]) / 10;
				params[4] = params[5];
				params[2] = paramsTemp[0];
				params[5] = paramsTemp[1];
				m_featureMapHisto->setMatrix(Pose::eulerToMat(params.data()));
				FeatureMapsRegistrationAlgorithm regAlg(*m_featureMapVol, *featureMapHistoVol);
				regAlg.p_rotationRange = p_disaRotationRange.value();
				regAlg.p_translationRange = p_disaTranslationRange.value();
				regAlg.compute();
				regAlg.p_transformation = FeatureMapsRegistrationAlgorithm::Transformation::Affine;
				regAlg.compute();
				if (regAlg.bestVal() > currentMax)
				{
					currentMax = regAlg.bestVal();
					optHistoMat = m_featureMapHisto->matrixToWorld();
				}
			}
			featureMapHistoVol->setMatrixToWorld(optHistoMat);
			m_sliceMat = optHistoMat * (mat4() << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1).finished();

			sliceDescWorld.setMatrixToWorld(m_sliceMat);
			auto slice = IM::makeArray(*m_preprocessedVol /*, false, IM::MagFilter::Linear, IM::Wrap::ClampToBorder, vec4f::Zero()*/)
							 .resample(m_preprocessedVol->get()->descriptorWorld(), sliceDescWorld)
							 .evaluateIntoImage<float>(true);
			m_preprocessedHisto->setMatrix(mat4::Identity());
			m_imgReg = std::make_unique<ImageRegistration>(slice.get(), m_preprocessedHisto.get());
			m_imgReg->setOptimizer(createOptimizer());                                 // set to OptimizerNL
			dynamic_cast<OptimizerNL*>(m_imgReg->optimizer().get())->setStep(15.0);    // Soft tissue block registration
			m_imgReg->optimizer()->setBounds(std::vector<double>({100, 100, 200}));    // bound for soft tissue block registration
			m_imgReg->optimizer()->setSelection(std::vector<bool>({0, 0, 1}));         // rotation

			m_imgReg->parametric()->setSimilarityMeasure(SimilarityMeasureFactory::Mode::NCC);    // set similarity measure
			m_imgReg->setTransformationModel(ImageRegistration::TransformationModel::Linear);
			m_imgReg->compute();    // perform registration

			dynamic_cast<OptimizerNL*>(m_imgReg->optimizer().get())->setStep(10.0);
			m_imgReg->optimizer()->setSelection(std::vector<bool>({1, 1, 0, 0, 0, 0}));    // translation
			m_imgReg->compute();

			m_imgReg->optimizer()->setSelection(std::vector<bool>({0, 0, 1, 0, 0, 0}));    // rotation
			m_imgReg->compute();

			m_imgReg->optimizer()->setSelection(std::vector<bool>({1, 1, 0, 0, 0, 0}));    // translation
			m_imgReg->compute();                                                           // perform registration

			m_imgReg->parametric()->setAffine(true);    // set to affine registration for soft tissue block registration
			dynamic_cast<OptimizerNL*>(m_imgReg->optimizer().get())->setStep(1.0);
			m_imgReg->parametric()->setSimilarityMeasure(SimilarityMeasureFactory::Mode::LNCC);
			m_imgReg->optimizer()->setSelection(std::vector<bool>({0, 0, 0, 1, 1, 0}));    // scale x/y
			m_imgReg->compute();                                                           // perform registration

			m_imgReg->parametric()->setAffine(false);
			m_imgReg->setTransformationModel(ImageRegistration::TransformationModel::FFD);
			dynamic_cast<FFDImageRegistration*>(m_imgReg->parametric())->setSmoothness(1e-3);
			m_imgReg->optimizer()->setDimension(32);    // 16 control points * (x,y) coordinates
			m_imgReg->optimizer()->setSelection(std::vector<bool>(m_imgReg->optimizer()->dimension(), 1));
			m_imgReg->compute();
			m_2dRegMat = m_preprocessedHisto->matrixToWorld().inverse();
		}

		OwningDataList HistoToCTRegistrationAlgorithm::takeOutput()
		{
			OwningDataList dl;

			if (m_preprocessedHisto)
				dl.add(std::move(m_preprocessedHisto));
			if (m_optimalSlice1)
				dl.add(std::move(m_optimalSlice1));
			if (m_optimalSlice2)
				dl.add(std::move(m_optimalSlice2));

			return dl;
		}
	}
}
