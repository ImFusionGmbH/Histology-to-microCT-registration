/* Copyright (c) 2012-2025 ImFusion GmbH, Munich, Germany. All rights reserved. */
#pragma once

#include <ImFusion/Base/Algorithm.h>
#include <ImFusion/Base/CostFunction.h>
#include <ImFusion/Core/Geometry/Plane.h>
#include <ImFusion/Core/Parameter.h>
#include <ImFusion/Core/SubProperty.h>
#include <ImFusion/GL/SharedImageSet.h>
#include <ImFusion/Histology/HistologyMicroCTPreprocessingAlgorithm.h>
#include <ImFusion/Reg/SimilarityMeasureWrapper.h>

namespace ImFusion
{
	class ImageRegistration;

	namespace Histology
	{
		class TiledImageSet;

		/// Class for computing 2D-3D registration between 2D histology slide and 3D micro-CT volume
		class HistoToCTRegistrationAlgorithm : public Algorithm
		{
		public:
			enum class InitMode
			{
				DISA = 0,     ///< DISA initialization
				MANUAL = 1    ///< Mannual initialization
			};

			/// \name	Algorithm interface methods
			///\{
			/// Create algorithm instance
			static bool createCompatible(const DataList& data, Algorithm** a = 0);

			/// Constructor, reference to the histology image and the CT volume required
			HistoToCTRegistrationAlgorithm(TiledImageSet& tis, SharedImageSet& vol);
			/// Overloaded constructor for a 2D SharedImageSet as input histology image
			HistoToCTRegistrationAlgorithm(SharedImageSet& histo, SharedImageSet& vol);

			/// Destructor
			~HistoToCTRegistrationAlgorithm();

			/// Compute 2D-3D registration
			void compute() override;

			OwningDataList takeOutput() override;

			/// Setter and getter for matrix to extract the CT slice from the volume
			void setSliceMat(mat4 mat) { m_sliceMat = mat; };
			mat4 sliceMat() { return m_sliceMat; }

			/// The pyramid level of histology slide
			Parameter<int> p_level = {"level", 2, *this};

			/// Search range of the CT slice within the volume, only for rough initialization. The unit is in pixel
			Parameter<std::vector<int>> p_sliceBound = {"sliceBound", std::vector<int>{100, 150}, *this};

			/// Mode of initialization
			Parameter<InitMode> p_initMode = {"initMode", InitMode::DISA, *this};

			/// Number of random search
			Parameter<int> p_numRandomSearch = {"numRandomSearch", 10, *this};

			/// Range of translation for DISA initialization
			Parameter<double> p_disaTranslationRange = {"disaTranslationRange", 40.0, *this};

			/// Range of rotation for DISA initialization
			Parameter<double> p_disaRotationRange = {"disaRotationRange", 10.0, *this};

			/// Image preprocessing algorithm for histology and micro-CT
			SubProperty<std::unique_ptr<HistologyMicroCTPreprocessingAlgorithm>> p_preprocessingAlgorithm = {"preprocessingAlgorithm", nullptr, this};

		private:
			/// Compute initialization
			void planeInitialization();

			/// The actual computation of DISA initialization
			void computeDISAInit();

			/// Compute feature maps using DISA 2D model
			void computeFeatureMap(SharedImageSet& histoSlide, SharedImageSet& vol);

			/// Compute refined 2D-3D registration after initialization
			void compute2D3DRegistration();

			TiledImageSet* m_tis = nullptr;                         // Input histology slide
			SharedImageSet* m_histo = nullptr;                      // (Alternative) Input histology slide
			SharedImageSet* m_vol;                                  // Input micro-CT volume
			std::unique_ptr<SharedImageSet> m_preprocessedVol;      // Preprocessed input micro-CT volume
			std::unique_ptr<SharedImageSet> m_preprocessedHisto;    // Preprocessed input histology slide
			std::unique_ptr<SharedImageSet> m_featureMapVol;        // Feature map of micro-CT volume
			std::unique_ptr<SharedImageSet> m_featureMapHisto;      // Feature map of histology slide
			std::unique_ptr<SharedImageSet> m_optimalSlice1;        // Registered CT slice with no out-of-plane deformation
			std::unique_ptr<SharedImageSet> m_optimalSlice2;        // Registered CT slice with out-of-plane deformation
			std::unique_ptr<ImageRegistration> m_imgReg;            // Regular image registration algorithm for intermediate 2D-2D registration
			mat4 m_sliceMat = mat4::Zero();                         // Matrix to extract the CT slice from the volume
			mat4 m_2dRegMat = mat4::Identity();                     // 2D-2D registration matrix between extracted 2D CT slice and histology slide
		};
	}
}
