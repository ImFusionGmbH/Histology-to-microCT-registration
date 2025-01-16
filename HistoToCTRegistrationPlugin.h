#pragma once

#include <ImFusion/Base/AlgorithmControllerFactory.h>
#include <ImFusion/Base/AlgorithmFactory.h>
#include <ImFusion/Base/ImFusionPlugin.h>

namespace ImFusion
{
	class Algorithm;
	class AlgorithmController;

	class HistoToCTRegistrationAlgorithmFactory : public AlgorithmFactory
	{
	public:
		HistoToCTRegistrationAlgorithmFactory();
	};

	class HistoToCTRegistrationControllerFactory : public AlgorithmControllerFactory
	{
	public:
		AlgorithmController* create(Algorithm* a) const override;
	};

	class HistoToCTRegistrationPlugin : public ImFusionPlugin
	{
	public:
		HistoToCTRegistrationPlugin();
		~HistoToCTRegistrationPlugin() override;

		const AlgorithmFactory* getAlgorithmFactory() override;

		const AlgorithmControllerFactory* getAlgorithmControllerFactory() override;

	private:
		AlgorithmFactory* m_algFactory;
		AlgorithmControllerFactory* m_algCtrlFactory;
	};

}
