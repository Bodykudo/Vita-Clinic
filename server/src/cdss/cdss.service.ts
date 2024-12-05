import {
  ConflictException,
  Injectable,
  NotFoundException,
  UnprocessableEntityException,
} from '@nestjs/common';
import { PrismaService } from 'src/prisma.service';
import {
  ApprovePredictionDto,
  CreatePredictionDto,
  PredictionDto,
  RejectPredictionDto,
  UpdatePredictionResultDto,
} from './dto/cdss.dto';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { NotificationsService } from 'src/notifications/notifications.service';

@Injectable()
export class CdssService {
  constructor(
    private prisma: PrismaService,
    private readonly httpService: HttpService,
    private notificationsService: NotificationsService,
  ) {}

  async getPrediction(
    predictionId: string,
    userId: string,
  ): Promise<PredictionDto> {
    return this.prisma.prediction.findUnique({
      where: { id: predictionId, userId },
      include: {
        instance: {
          include: {
            series: {
              include: {
                study: {
                  include: {
                    scan: true,
                  },
                },
              },
            },
          },
        },
      },
    });
  }

  async create(
    userId: string,
    createPredictionDto: CreatePredictionDto,
  ): Promise<PredictionDto> {
    const instance = await this.prisma.instance.findUnique({
      where: { id: createPredictionDto.instanceId },
    });

    if (!instance) {
      throw new NotFoundException('Instance not found');
    }

    const prediction = await this.prisma.prediction.create({
      data: { ...createPredictionDto, userId },
    });

    let url = `${process.env.CDSS_URL}`;
    switch (createPredictionDto.model) {
      case 'brain_mri':
        url += '/inference/brain-tumors-classification';
        break;
      case 'lung_ct':
        url += '/inference/chest-ct-cancer-classification';
        break;
      case 'mammography':
        url += '/inference/mammography-cancer-classification';
        break;
      default:
        throw new UnprocessableEntityException('Invalid prediction model');
    }
    const headers = { 'X-API-KEY': process.env.CDSS_API_KEY };

    try {
      await firstValueFrom(
        this.httpService.post(
          url,
          {
            predictionId: prediction.id,
            instance: instance.url,
          },
          { headers },
        ),
      );
      return prediction;
    } catch (error) {
      await this.prisma.prediction.delete({ where: { id: prediction.id } });
      throw new Error(`Error creating prediction: ${error.message}`);
    }
  }

  async updatePredictionResult(
    predictionId: string,
    updatePredictionResultDto: UpdatePredictionResultDto,
  ): Promise<PredictionDto> {
    const prediction = await this.prisma.prediction.findUnique({
      where: { id: predictionId },
      include: {
        instance: {
          include: {
            series: {
              include: {
                study: {
                  include: {
                    scan: true,
                  },
                },
              },
            },
          },
        },
      },
    });

    if (!prediction) {
      throw new NotFoundException('Prediction not found');
    }

    if (prediction.status !== 'pending') {
      throw new ConflictException('Prediction is not pending');
    }

    const updatedPrediction = await this.prisma.prediction.update({
      where: { id: predictionId },
      data: { ...updatePredictionResultDto, status: 'predicted' },
    });

    await this.notificationsService.create({
      userId: prediction.userId,
      targetId: prediction.id,
      targetName: `Scan: ${prediction.instance.series.study.scan.title}, Series: ${prediction.instance.series.seriesNumber}, Instance: ${prediction.instance.instanceNumber}`,
      type: 'ai_result',
    });

    return updatedPrediction;
  }

  async failPrediction(predictionId: string): Promise<PredictionDto> {
    const prediction = await this.prisma.prediction.findUnique({
      where: { id: predictionId },
      include: {
        instance: {
          include: {
            series: {
              include: {
                study: {
                  include: {
                    scan: true,
                  },
                },
              },
            },
          },
        },
      },
    });

    if (!prediction) {
      throw new NotFoundException('Prediction not found');
    }

    if (prediction.status !== 'pending') {
      throw new ConflictException('Prediction is not pending');
    }

    const updatedPrediction = await this.prisma.prediction.update({
      where: { id: predictionId },
      data: { status: 'failed' },
    });

    await this.notificationsService.create({
      userId: prediction.userId,
      targetId: prediction.id,
      targetName: `Scan: ${prediction.instance.series.study.scan.title}, Series: ${prediction.instance.series.seriesNumber}, Instance: ${prediction.instance.instanceNumber}`,
      type: 'ai_failed',
    });

    return updatedPrediction;
  }

  async approvePrediction(
    userId: string,
    predictionId: string,
    approvePredictionDto: ApprovePredictionDto,
  ): Promise<PredictionDto> {
    const prediction = await this.prisma.prediction.findUnique({
      where: { id: predictionId, userId },
      include: {
        instance: {
          include: {
            series: {
              include: {
                study: {
                  include: {
                    scan: true,
                  },
                },
              },
            },
          },
        },
      },
    });

    if (!prediction) {
      throw new NotFoundException('Prediction not found');
    }

    if (prediction.status !== 'predicted') {
      throw new ConflictException('Prediction is not predicted');
    }

    const updatedPrediction = await this.prisma.prediction.update({
      where: { id: predictionId },
      data: {
        status: 'approved',
        ...approvePredictionDto,
      },
    });

    return updatedPrediction;
  }

  async rejectPrediction(
    userId,
    predictionId: string,
    rejectedPredictionDto: RejectPredictionDto,
  ): Promise<PredictionDto> {
    const prediction = await this.prisma.prediction.findUnique({
      where: { id: predictionId, userId },
      include: {
        instance: {
          include: {
            series: {
              include: {
                study: {
                  include: {
                    scan: true,
                  },
                },
              },
            },
          },
        },
      },
    });

    if (!prediction) {
      throw new NotFoundException('Prediction not found');
    }

    if (prediction.status !== 'predicted') {
      throw new ConflictException('Prediction is not predicted');
    }

    const updatedPrediction = await this.prisma.prediction.update({
      where: { id: predictionId },
      data: {
        status: 'rejected',
        ...rejectedPredictionDto,
      },
    });

    return updatedPrediction;
  }
}
